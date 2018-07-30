#!/usr/bin/env python
import argparse
import logging
import os
import csv
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from data import Corpus
from model import RNNG
from util import Timer, get_subdir_string

parser = argparse.ArgumentParser(description='Discriminative RNNG parser')
parser.add_argument('mode', choices=['dev', 'test', 'train', 'parse'],
                    help='type')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='print out parser states')
parser.add_argument('--data', type=str, default='../tmp',
                    help='location of the data corpus')
parser.add_argument('--textline', type=str, choices=['unked', 'lower', 'upper'], default='unked',
                    help='textline to use from the oracle file')
parser.add_argument('--outdir', type=str, default='',
                    help='location to make output log and checkpoint folders')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=5.,
                    help='clipping gradient norm at this value')
parser.add_argument('--print_every', type=int, default=10,
                    help='when to print training progress')
parser.add_argument('--use_glove', action='store_true',
                    help='using pretrained glove embeddings')
parser.add_argument('--char', action='store_true',
                    help='using pretrained glove embeddings')
args = parser.parse_args()

# Create folders for logging and checkpoints
SUBDIR = get_subdir_string(args)
LOGDIR = os.path.join(args.outdir, 'log', SUBDIR)
CHECKDIR = os.path.join(args.outdir, 'checkpoints')
# CHECKDIR = os.path.join(args.outdir, 'checkpoints', SUBDIR)
LOGFILE = os.path.join(LOGDIR, 'train.log')
CHECKFILE = os.path.join(CHECKDIR, 'model.pt')
os.mkdir(LOGDIR)
if not os.path.exists(CHECKDIR):
    os.mkdir(CHECKDIR)

torch.manual_seed(42)

corpus = Corpus(data_path=args.data, textline=args.textline, char=args.char)
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
             use_cuda=False,
             use_glove=args.use_glove,
             char=args.char)

parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr=args.lr)

logfile = open(LOGFILE, 'w')
trainfile = open(LOGFILE + '.train.txt', 'w')
parsefile = open(LOGFILE + '.parse.txt', 'w')

if args.mode == 'dev':
    sent, indices, actions = train_batches[0]
    loss = model(sent, indices, actions, verbose=args.verbose)

if args.mode == 'test':
    sent, indices, actions = train_batches[0]
    timer = Timer()
    model.train()
    try:
        for step in range(100):
            loss = model(sent, indices, actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()

            time = timer.elapsed()
            if step % args.print_every == 0:
                print('Step {} | loss {:.3f} | {:.3f}s/sent '.format(step, loss.data[0], args.print_every/time))
    except KeyboardInterrupt:
        print('Exiting training early.')

    model.eval()
    loss = model(sent, indices, actions, verbose=args.verbose, file=trainfile)
    parser = model.parse(sent, indices, verbose=args.verbose, file=parsefile)
    torch.save(model, CHECKFILE)

    print('Finished parsing.', file=parsefile)
    print(parser, file=parsefile)

if args.mode == 'train':
    losses = []
    timer = Timer()
    try:
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
                avg_loss = np.mean(losses[:-args.print_every])
                print('Step {} | loss {:.3f} | {:.3f} sents/sec '.format(step, avg_loss, args.print_every/time))

    except KeyboardInterrupt:
        print('Exiting training early.')

    model.eval()
    parser = model.parse(sent, indices, verbose=args.verbose, file=parsefile)
    torch.save(model, CHECKFILE)

    print('Finished parsing.', file=parsefile)
    print(parser, file=parsefile)

if args.mode == 'parse':
    model = torch.load(CHECKFILE)
    sent, actions = next(train_batches)
    parser = model.parse(sent, file=logfile)
    print(parser)
    print('ACTIONS')
    print([model.dictionary.i2a[i] for i in actions])
