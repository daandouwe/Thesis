#!/usr/bin/env python
import os
import sys
import time
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

from data import Corpus
from model import DiscRNNG, make_model
from decode import GreedyDecoder
from eval import evalb
from utils import Timer, get_subdir_string, get_folders, write_args


def clock_time(s):
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return int(h), int(m), int(s)


def worker(rank, size, model, batches, optimizer, return_dict, elbo_objective, print_every=10):
    """Distributed Synchronous SGD."""
    # Restrict each processor to use only 1 thread.
    torch.set_num_threads(1)  # Without this it won't work on Lisa (and slow down on laptop)!
    chunk_size = len(batches) // dist.get_world_size()
    start = rank * chunk_size
    stop = (rank+1) * chunk_size
    num_steps = len(batches) // size
    losses = []
    start_time = time.time()
    t0 = time.time()
    for i, batch in enumerate(batches[start:stop], 1):
        sentence, actions = batch
        loss = model(sentence, actions)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        average_gradients(model)
        optimizer.step()

        # Compute the average loss for logging.
        dist.all_reduce(loss, op=dist.reduce_op.SUM)  # inplace operation
        losses.append(loss.data.item() / size)

        if i % print_every == 0 and rank == 0:
            sents_per_sec = print_every * size / (time.time() - t0)
            t0 = time.time()
            message = 'step {}/{} ({:.0f}%) | loss {:.3f} | {:.3f} sents/sec | elapsed {}h{:02}m{:02}s | eta {}h{:02}m{:02}s'.format(
                    i, num_steps, i/num_steps * 100,
                    np.mean(losses[-print_every:]),
                    sents_per_sec,
                    *clock_time(time.time() - start_time),
                    *clock_time((len(batches) - i*size) / sents_per_sec))
            if elbo_objective:
                message += (f'| alpha {model.criterion.annealer._alpha:.3f} ' +
                    f'| temp {model.stack.encoder.composition.annealer._temp:.3f} ')
            print(message)
            # Save model every now and then.
            return_dict['model'] = model
    # Save model when finished.
    if rank == 0:
        return_dict['model'] = model


def average_gradients(model):
    """Gradient averaging."""
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:  # some layers of model are not used and have no grad
        # if param is not None and param.requires_grad:  # some layers of model are not used and have no grad
            # print(param)
            # dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            # param.grad.data /= size
            dist.all_reduce(param.grad, op=dist.reduce_op.SUM)
            param.grad /= size


def init_processes(fn, *args, backend='tcp'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    rank, size, *rest = args
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(*args)


def train_epoch(size, model, train_batches, optimizer, elbo_objective):
    try:
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        for rank in range(size):
            p = mp.Process(
                target=init_processes,
                args=(worker, rank, size, model, train_batches, optimizer, return_dict, elbo_objective)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        # Close open processes.
        for p in processes:
            p.join()
        print('-'*89)
        print('Exiting from training early.')
    # Catch trained model.
    return return_dict['model']


def main(args):

    size = mp.cpu_count() if args.nprocs == -1 else args.nprocs
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make output folder structure.
    if not args.disable_folders:
        subdir, logdir, checkdir, outdir = get_folders(args)
        args.logdir, args.checkdir, args.outdir = logdir, checkdir, outdir
        os.mkdir(logdir); os.mkdir(checkdir); os.mkdir(outdir)
        print(f'Output subdirectory: `{subdir}`.')
    else:
        print('Did not make output folders!')

    # Save arguments.
    write_args(args, logdir)

    # Initialize model and data.
    corpus = Corpus(
        data_path=args.data,
        model=args.model,
        textline=args.textline,
        name=args.name,
        use_chars=args.use_chars,
        max_lines=args.max_lines
    )
    train_batches = corpus.train.batches(length_ordered=False, shuffle=False)
    dev_batches = corpus.dev.batches(length_ordered=False, shuffle=False)[:100]
    test_batches = corpus.test.batches(length_ordered=False, shuffle=False)[:100]
    print(corpus)

    model = make_model(args, corpus.dictionary)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    dev_pred_path = os.path.join(args.outdir, f'{args.name}.dev.pred.trees')
    dev_gold_path = os.path.join(args.data, 'dev', f'{args.name}.dev.trees')
    dev_result_path = os.path.join(args.outdir, f'{args.name}.dev.result')


    print(f'Training with {size} processes...')
    losses = list()
    num_updates = 0
    best_dev_fscore = -np.inf
    best_dev_epoch = None
    test_fscore = None
    checkfile = os.path.join(checkdir, 'model.pt')

    def save_checkpoint():
        with open(checkfile, 'wb') as f:
            state = {
                'args': args,
                'model': model,
                'dictionary': corpus.dictionary,
                'optimizer': optimizer,
                'scheduler': None,
                'epoch': epoch,
                'num-updates': num_updates,
                'best-dev-fscore': best_dev_fscore,
                'best-dev-epoch': best_dev_epoch,
                'test-fscore': test_fscore
            }
            torch.save(state, f)

    def predict(batches):
        trees = []
        decoder = GreedyDecoder(
            model=model, dictionary=corpus.dictionary, use_chars=args.use_chars)
        for i, batch in enumerate(batches):
            sentence, actions = batch
            tree, *rest = decoder(sentence)
            trees.append(tree)
            if i % 10 == 0:
                print(f'Predicting sentence {i}/{len(batches)}...', end='\r')
        return trees

    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_epoch

        model.eval()
        # Predict trees.
        trees = predict(dev_batches)
        with open(dev_pred_path, 'w') as f:
            print('\n'.join([tree.linearize() for tree in trees]), file=f)
        # Compute f-score.
        dev_fscore = evalb(args.evalb_dir, dev_pred_path, dev_gold_path, dev_result_path)
        # Log score to tensorboard.
        if dev_fscore > best_dev_fscore:
            print(f'Saving new best model to `{checkfile}`...')
            save_checkpoint()
            best_dev_epoch = epoch
            best_dev_fscore = dev_fscore
        return dev_fscore


    elbo_objective = (args.composition in ('latent-factors', 'latent-attention'))
    print(f'ELBO objective: {elbo_objective}.')

    epoch_timer = Timer()
    for epoch in range(1, args.epochs+1):
        # Shuffle batches each epoch.
        np.random.shuffle(train_batches)

        # Train one epoch.
        model = train_epoch(size, model, train_batches, optimizer, elbo_objective)

        dev_fscore = check_dev()

        print('-'*99)
        print(
            f'| End of epoch {epoch:3d}/{args.epochs} '
            f'| elapsed {epoch_timer.format_elapsed()} '
            f'| dev-fscore {dev_fscore:4.2f} '
            f'| best dev-epoch {best_dev_epoch} '
            f'| best dev-fscore {best_dev_fscore:4.2f} '
        )
        print('-'*99)

    # Load best saved model.
    print(f'Loading best saved model (epoch {best_dev_epoch}) from `{checkfile}`...')
    with open(checkfile, 'rb') as f:
        state = torch.load(f)
        model = state['model']

    print('Evaluating loaded model on test set...')
    test_pred_path = os.path.join(args.outdir, f'{args.name}.test.pred.trees')
    test_gold_path = os.path.join(args.data, 'test', f'{args.name}.test.trees')
    test_result_path = os.path.join(args.outdir, f'{args.name}.test.result')
    trees = predict(test_batches)
    with open(test_pred_path, 'w') as f:
        print('\n'.join([tree.linearize() for tree in trees]), file=f)
    test_fscore = evalb(args.evalb_dir, test_pred_path, test_gold_path, test_result_path)
    save_checkpoint()

    print('-'*99)
    print(
         f'| End of training '
         f'| best dev-epoch {best_dev_epoch:2d} '
         f'| best dev-fscore {best_dev_fscore:4.2f} '
         f'| test-fscore {test_fscore}'
    )
    print('-'*99)
