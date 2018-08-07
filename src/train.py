import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from data import Corpus
from model import make_model
from predict import predict, write_prediction
from eval import eval
from util import Timer, clock_time, get_subdir_string, write_losses, make_folders

LOSSES = []

def schedule_lr(args, optimizer, update):
    update = update + 1
    warmup_coeff = args.lr / args.learning_rate_warmup_steps
    if update <= args.learning_rate_warmup_steps:
        for param_group in optimizer.param_groups:
            param_group['lr'] = update * warmup_coeff

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def batchify(batches, batch_size):
    div = lambda a, b : ((a-1) // b) + 1
    return [batches[i*batch_size:(i+1)*batch_size]
                for i in range(div(len(batches), batch_size))]

def train(args, model, batches, optimizer, epoch):
    """One epoch of training."""
    model.train()
    timer = Timer()
    num_batches = len(batches) // args.batch_size
    for step, minibatch in enumerate(batchify(batches, args.batch_size), 1):
        loss = torch.zeros(1, device=args.device)
        for batch in minibatch:
            sentence, actions = batch
            loss += model(sentence, actions)
        loss /= args.batch_size

        schedule_lr(args, optimizer, step)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        loss = loss.item()
        LOSSES.append(loss)
        if step % args.print_every == 0:
            sents_per_sec = args.batch_size * args.print_every / timer.elapsed()
            avg_loss = np.mean(LOSSES[-args.print_every:])
            eta = clock_time((len(batches) - step) / sents_per_sec)
            print(
                f'| step {step:6d}/{num_batches:5d} '
                f'| loss {avg_loss:3.3f} '
                f'| {sents_per_sec:4.1f} sents/sec '
                f'| lr {get_lr(optimizer):.1e} '
                 '| eta {:2d}h:{:2d}m:{:2d}s |'.format(*eta)
            )

def evaluate(model, batches):
    model.eval()
    with torch.no_grad(): # operations inside don't track history
        total_loss = 0.
        for step, batch in enumerate(batches, 1):
            # sent, indices, actions = batch
            sentence, actions = batch
            # loss = model(sent, indices, actions)
            loss = model(sentence, actions)
            total_loss += loss.item()
    return total_loss / step

def main(args):
    # Set random seed.
    torch.manual_seed(args.seed)
    # Set cuda.
    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Device: {args.device}.')

    if not args.disable_folders:
        make_folders(args)

    print(f'Loading data from {args.data}...')
    corpus = Corpus(data_path=args.data, textline=args.textline, char=args.use_char)
    train_batches = corpus.train.batches(length_ordered=False, shuffle=False)
    dev_batches   = corpus.dev.batches(length_ordered=False, shuffle=False)
    test_batches  = corpus.test.batches(length_ordered=False, shuffle=False)
    print(corpus)

    model = make_model(args, corpus.dictionary)
    model.to(args.device)

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=1., betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max',
        factor=args.step_decay_factor,
        patience=args.step_decay_patience,
        verbose=True,
    )

    best_dev_epoch = None
    best_dev_loss = None

    print('Start training.')
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            train(args, model, train_batches, optimizer, epoch)
            dev_loss = evaluate(model, dev_batches)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_dev_loss or dev_loss < best_dev_loss:
                with open(args.checkfile, 'wb') as f:
                    torch.save(model, f)
                best_dev_epoch = epoch
                best_dev_loss = dev_loss
            print('-'*89)
            print(
                f'| End of epoch {epoch:3d}/{args.epochs:3d} '
                f'| dev loss {dev_loss:2.4f} '
                f'| best dev epoch {best_dev_epoch:2d} '
                f'| best dev loss {best_dev_loss:2.4f} '
            )
            print('-'*89)
    except KeyboardInterrupt:
        print('-'*89)
        print('Exiting from training early.')
        # Save the losses for plotting and diagnostics.
        write_losses(args, LOSSES)
        print('Evaluating on development set...')
        dev_loss = evaluate(model, dev_batches)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_dev_loss or dev_loss < best_dev_loss:
            with open(args.checkfile, 'wb') as f:
                torch.save(model, f)
            best_dev_epoch = epoch
            best_dev_loss = dev_loss

    # Load best saved model.
    with open(args.checkfile, 'rb') as f:
        model = torch.load(f)

    # Evaluate on test set.
    print('Evaluating on test set...')
    test_loss = evaluate(model, test_batches)
    print('-'*89)
    print('| End of training | test loss {:2.4f} | best dev epoch {:2d} | best dev loss {:2.4f}'.format(
            test_loss, best_dev_epoch, best_dev_loss))
    print('-'*89)

    # Predict with best model on test set.
    print('Predicting on test set...')
    predict(args, model, test_batches, name='test')
    print('Evaluating test set predictions...')
    eval(args.outdir)
    print('Done.')
