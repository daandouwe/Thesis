import logging
import os

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

def train(args, model, batches, optimizer):
    """One epoch of training."""
    model.train()
    timer = Timer()
    num_batches = len(batches)
    for step, batch in enumerate(batches, 1):
        # sent, indices, actions = batch
        sentence, actions = batch
        # Idea: make sentence and actions a list of Vocab items:
        # word = vocab.word, index = vocab.index
        # action = vocab.word, action_index = vocab.index https://github.com/nikitakit/self-attentive-parser/blob/master/src/vocabulary.py
        # loss = model(sentence, indices, actions)
        loss = model(sentence, actions)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        loss = loss.item()
        LOSSES.append(loss)
        if step % args.print_every == 0:
            sents_per_sec = args.print_every / timer.elapsed()
            avg_loss = np.mean(LOSSES[-args.print_every:])
            eta = clock_time((num_batches - step) / sents_per_sec)
            print('| step {:6d}/{:5d} | loss {:3.4f} | {:4.3f} sents/sec | eta {:2d}h:{:2d}m:{:2d}s'.format(
                        step, num_batches, avg_loss, sents_per_sec, *eta))

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
    print('Device: {}.'.format(args.device))

    if not args.disable_folders:
        make_folders(args)

    corpus = Corpus(data_path=args.data, textline=args.textline, char=args.use_char)
    train_batches = corpus.train.batches(length_ordered=False, shuffle=False)
    dev_batches   = corpus.dev.batches(length_ordered=False, shuffle=False)
    test_batches  = corpus.test.batches(length_ordered=False, shuffle=False)
    print(corpus)

    model = make_model(args, corpus.dictionary)
    model.to(args.device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # NOTE Use L2 regularization yarin gal's paper page 9 for formula.
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
                with open(args.checkfile, 'wb') as f:
                    torch.save(model, f)
                best_dev_epoch = epoch
                best_dev_loss = dev_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
                parameters = filter(lambda p: p.requires_grad, model.parameters())
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
            with open(args.checkfile, 'wb') as f:
                torch.save(model, f)
            best_dev_epoch = epoch
            best_dev_loss = dev_loss

    # Save the losses for plotting and diagnostics.
    write_losses(args, LOSSES)

    # Load best saved model.
    with open(args.checkfile, 'rb') as f:
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
    eval(args.outdir)
