import argparse
import logging
import os
import csv
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from get_vocab import get_sentences
from data import Corpus, load_glove
from model import RNNG
from utils import Timer, get_subdir_string

parser = argparse.ArgumentParser(description='Discriminative RNNG parser')
parser.add_argument('--data', type=str, default='../tmp',
                    help='location of the data corpus')
parser.add_argument('--textline', type=str, choices=['unked', 'lower', 'upper'], default='unked',
                    help='textline to use from the oracle file')
parser.add_argument('--outdir', type=str, default='',
                    help='location to make output log and checkpoint folders')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs')
parser.add_argument('--clip', type=float, default=5.,
                    help='clipping gradient norm at this value')
parser.add_argument('--print_every', type=int, default=10,
                    help='when to print training progress')
parser.add_argument('--use_glove', type=bool, default=False,
                    help='using pretrained glove embeddings')
args = parser.parse_args()

torch.manual_seed(42)

# Create folders for logging and checkpoints
SUBDIR = get_subdir_string(args)
LOGDIR = os.path.join(args.outdir, 'log', SUBDIR)
CHECKDIR = os.path.join(args.outdir, 'checkpoints', SUBDIR)
# OUTDIR = os.path.join(args.outdir, 'out', SUBDIR)
OUTDIR = 'out'
os.mkdir(LOGDIR)
os.mkdir(CHECKDIR)
# os.mkdir(OUTDIR)
LOGFILE = os.path.join(LOGDIR, 'train.log')
CHECKFILE = os.path.join(CHECKDIR, 'model.pt')
# OUTFILE = os.path.join(OUTDIR, 'train.predict.txt')

def train():
    """One epoch of training."""
    model.train()
    for step, batch in enumerate(train_batches, 1):
        sent, indices, actions = batch
        loss = model(sent, indices, actions)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        loss = loss.data[0]
        losses.append(loss)
        if step % args.print_every == 0:
            time = timer.elapsed()
            avg_loss = np.mean(losses[-args.print_every:])
            print('Step {}/{} | loss {:.3f} | {:.3f} sents/sec '.format(step, num_batches, avg_loss, args.print_every/time))

def evaluate():
    model.eval()
    dev_loss = 0.
    for step, batch in enumerate(dev_batches, 1):
        sent, indices, actions = batch
        loss = model(sent, indices, actions)
        dev_loss += loss.data[0]
    dev_loss /= step
    return dev_loss

def write_prediction(dev_sentences, test_sentences, outdir, verbose=False):
    def print_sent_dict(sent_dict, file):
        print(sent_dict['tree'], file=file)
        print(sent_dict['tags'], file=file)
        print(sent_dict['upper'], file=file)
        print(sent_dict['lower'], file=file)
        print(sent_dict['unked'], file=file)
        print('\n'.join(sent_dict['actions']), file=file)
        print(file=file)
    dev_path = os.path.join(outdir, 'dev.pred.oracle')
    test_path = os.path.join(outdir, 'test.pred.oracle')
    with open(dev_path, 'w') as f:
        for i, sent_dict in enumerate(dev_sentences):
            if verbose: print(i, end='\r')
            print_sent_dict(sent_dict, f)
    with open(test_path, 'w') as f:
        for i, sent_dict in enumerate(test_sentences):
            if verbose: print(i, end='\r')
            print_sent_dict(sent_dict, f)

def predict():
    model.eval()
    dev_sentences = get_sentences(os.path.join(args.data, 'dev', 'ptb.dev.oracle'))
    for i, batch in enumerate(dev_batches):
        sent, indices, actions = batch
        parser = model.parse(sent, indices)
        dev_sentences[i]['actions'] = parser.actions
    test_sentences = get_sentences(os.path.join(args.data, 'test', 'ptb.test.oracle'))
    for i, batch in enumerate(test_batches):
        sent, indices, actions = batch
        parser = model.parse(sent, indices)
        test_sentences[i]['actions'] = parser.actions
    write_prediction(dev_sentences, test_sentences, OUTDIR)


if __name__ == '__main__':
    corpus = Corpus(data_path=args.data, textline=args.textline)
    train_batches =  corpus.train.batches(length_ordered=False, shuffle=False)
    dev_batches   =  corpus.dev.batches(length_ordered=False, shuffle=False)
    test_batches  =  corpus.test.batches(length_ordered=False, shuffle=False)
    num_batches = len(train_batches)

    model = RNNG(dictionary=corpus.dictionary,
                 emb_dim=100,
                 emb_dropout=0.3,
                 lstm_hidden=100,
                 lstm_num_layers=1,
                 lstm_dropout=0.3,
                 mlp_hidden=500,
                 cuda=False,
                 use_glove=args.use_glove)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    try:
        losses = []
        timer = Timer()
        for epoch in range(args.epochs):
            train()
            dev_loss = evaluate()
            print('-'*79)
            print('End of epoch {}/{} | avg dev loss : {}'.format(epoch, args.epochs, dev_loss))
            print('')
            print('-'*79)

    except KeyboardInterrupt:
        print('Exiting training early.')

    print('Predicting.')
    predict()
    print('Finished')

    torch.save(model, CHECKFILE)
