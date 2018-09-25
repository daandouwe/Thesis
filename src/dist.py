#!/usr/bin/env python
import os
import sys
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import multiprocessing as mp
import numpy as np

from data import Corpus
from model import DiscRNNG, make_model
from predict import predict
from eval import true_evalb
from util import Timer, get_subdir_string, make_folders


def clock_time(s):
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return int(h), int(m), int(s)


def train(rank, size, model, batches, optimizer, return_dict, print_every=10):
    """Distributed Synchronous SGD Example."""
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
            print('step {}/{} ({:.0f}%) | loss {:.3f} | {:.3f} sents/sec | elapsed {}h{:02}m{:02}s '
                  '| eta {}h{:02}m{:02}s'.format(
                    i, num_steps, i/num_steps * 100,
                    np.mean(losses[-print_every:]),
                    sents_per_sec,
                    *clock_time(time.time() - start_time),
                    *clock_time((len(batches) - i*size) / sents_per_sec)))
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
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size


def init_processes(fn, *args, backend='tcp'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    rank, size, model, train_batches, optimizer, return_dict = args
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(*args)


def main(args):
    size = mp.cpu_count() if args.nprocs == -1 else args.nprocs
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    make_folders(args)

    # Initialize model and data.
    corpus = Corpus(data_path='../tmp', textline='unked', max_lines=args.max_lines)
    train_batches = corpus.train.batches(length_ordered=False, shuffle=False)
    dev_batches = corpus.dev.batches(length_ordered=False, shuffle=False)[:100]
    model = make_model(args, corpus.dictionary)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    try:
        print(f'Training with {size} processes...')
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        for rank in range(size):
            p = mp.Process(
                target=init_processes,
                args=(train, rank, size, model, train_batches, optimizer, return_dict)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    except:
        # Close open processes.
        for p in processes:
            p.join()
        print('-'*89)
        print('Exiting from training early.')

    model = return_dict['model']

    loss = 0
    for batch in train_batches[:10]:
        sentence, actions = batch
        loss += model(sentence, actions).item()
    loss /= 10
    print('Final:', loss)


    print(f'Evaluating model on development set to `{args.outdir}/{args.name}`...')
    model.eval()
    predict(model, dev_batches, args.outdir, name=args.name, set='dev')

    evalb_dir = os.path.expanduser(args.evalb_dir)
    pred_path = os.path.join(args.outdir, f'{args.name}.dev.pred.trees')
    gold_path = os.path.join(args.data, 'dev', f'{args.name}.dev.trees')
    result_path = os.path.join(args.outdir, f'{args.name}.result')
    true_evalb(evalb_dir, pred_path, gold_path, result_path)

    with open(pred_path) as f:
        print(f.read())
    print()
    with open(result_path) as f:
        print(f.read())

    with open(args.checkfile, 'wb') as f:
        state = {
            'args': args,
            'model': model,
            'dictionary': corpus.dictionary,
            'optimizer': optimizer,
            'epoch': 1,
            'num-updates': 0,
            'best-dev-fscore': 0,
            'best-dev-epoch': 0,
            'test-fscore': 0
        }
        torch.save(state, f)
