import argparse
import logging
import os
import csv
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from data import Corpus, load_glove
from model import RNNG
from predict import predict, write_prediction
from util import Timer, clock_time, get_subdir_string, write_args, write_losses

LOSSES = []

def train(args, model, batches, optimizer):
    """One epoch of training."""
    model.train()
    timer = Timer()
    num_batches = len(batches)
    for step, batch in enumerate(batches, 1):
        sent, indices, actions = batch
        loss = model(sent, indices, actions)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        loss = loss.data[0]
        LOSSES.append(loss)
        if step % args.print_every == 0:
            sents_per_sec = args.print_every / timer.elapsed()
            avg_loss = np.mean(LOSSES[-args.print_every:])
            eta = clock_time((num_batches - step) / sents_per_sec)
            print('| step {:3d}/{:3d} | loss {:2.4f} | {:4.3f} sents/sec | eta {:2d}h:{:2d}m:{:2d}s'.format(
                        step, num_batches, avg_loss, sents_per_sec, *eta))

def evaluate(model, batches):
    model.eval()
    total_loss = 0.
    for step, batch in enumerate(batches, 1):
        sent, indices, actions = batch
        loss = model(sent, indices, actions)
        total_loss += loss.data[0]
    return total_loss / step

def main(args):
    # Set random seed.
    torch.manual_seed(args.seed)
    # Set cuda.
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    print('Using CUDA: {}'.format(args.cuda))

    # Create folders for logging and checkpoints
    subdir = get_subdir_string(args)
    logdir = os.path.join(args.outdir, 'log', subdir)
    checkdir = os.path.join(args.outdir, 'checkpoints', subdir)
    outdir = os.path.join(args.outdir, 'out', subdir)
    # outdir = 'out'
    os.mkdir(logdir)
    os.mkdir(checkdir)
    os.mkdir(outdir)
    logfile = os.path.join(logdir, 'train.log')
    checkfile = os.path.join(checkdir, 'model.pt')
    outfile = os.path.join(outdir, 'train.predict.txt')

    # Add folders and dirs to args
    args.outdir = outdir
    args.outfile = outfile
    args.logdir = logdir
    args.logfile = logfile
    args.checkdir = checkdir
    args.checkfile = checkfile
    write_args(args)

    corpus = Corpus(data_path=args.data, textline=args.textline)
    train_batches = corpus.train.batches(length_ordered=False, shuffle=False)
    dev_batches   = corpus.dev.batches(length_ordered=False, shuffle=False)
    test_batches  = corpus.test.batches(length_ordered=False, shuffle=False)

    model = RNNG(dictionary=corpus.dictionary,
                 emb_dim=args.emb_dim,
                 emb_dropout=args.dropout,
                 lstm_hidden=args.lstm_dim,
                 lstm_num_layers=args.lstm_num_layers,
                 lstm_dropout=args.dropout,
                 mlp_hidden=args.mlp_dim,
                 use_cuda=args.cuda,
                 use_glove=args.use_glove)
    if args.cuda:
        model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    lr = args.lr
    best_dev_epoch = None
    best_dev_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            train(args, model, train_batches, optimizer)
            dev_loss = evaluate(model, dev_batches)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_dev_loss or dev_loss < best_dev_loss:
                with open(checkfile, 'wb') as f:
                    torch.save(model, f)
                best_dev_epoch = epoch
                best_dev_loss = dev_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
                optimizer = torch.optim.Adam(parameters, lr=lr)
            print('-'*89)
            print('| End of epoch {:3d}/{:3d} | dev loss {:2.4f}| best dev epoch {:2d} | best dev loss {:2.4f} | lr {:2.4f}'.format(
                        epoch, args.epochs, dev_loss, best_dev_epoch, best_dev_loss, lr))
            print('-'*89)
    except KeyboardInterrupt:
        print('-'*89)
        print('Exiting from training early.')
        dev_loss = evaluate(model, dev_batches)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_dev_loss or dev_loss < best_dev_loss:
            with open(checkfile, 'wb') as f:
                torch.save(model, f)
            best_dev_epoch = epoch
            best_dev_loss = dev_loss

    write_losses(args, LOSSES)

    # Load best saved model.
    with open(checkfile, 'rb') as f:
        model = torch.load(f)

    # Evaluate on test set.
    test_loss = evaluate(model, test_batches)
    print('-'*89)
    print('| End of training | test loss {:2.4f} | best dev epoch {:2d} | best dev loss {:2.4f}'.format(
            test_loss, best_dev_epoch, best_dev_loss))
    print('-'*89)

    # Predict with best model on test set.
    predict(args, model, test_batches, name='test')
    print('Finished predicting')
