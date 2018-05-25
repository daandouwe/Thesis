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
from utils import Timer, get_subdir_string

parser = argparse.ArgumentParser(description='Discriminative RNNG parser')
parser.add_argument('mode', choices=['test', 'train', 'parse'],
                    help='type')
parser.add_argument('--data', type=str, default='../tmp/ptb',
                    help='location of the data corpus')
parser.add_argument('--outdir', type=str, default='',
                    help='location to make output log and checkpoint folders')
parser.add_argument('--lr', type=float, default=1e-1,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=5.,
                    help='clipping gradient norm at this value')
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

corpus = Corpus(data_path=args.data)
batches = corpus.train.batches(length_ordered=False, shuffle=False)

model = RNNG(stack_size=len(corpus.dictionary.s2i),
             action_size=len(corpus.dictionary.a2i),
             emb_dim=20, emb_dropout=0.3,
             lstm_hidden=20, lstm_num_layers=1, lstm_dropout=0.3,
             mlp_hidden=50, cuda=False)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

logfile = open(LOGFILE, 'w')

if args.mode == 'test':
    sent, actions = next(batches)
    loss = model(sent, actions, corpus.dictionary, verbose=True)

if args.mode == 'train':
    sent, actions = next(batches)
    try:
        for step in range(100):
            # sent, actions = next(batches)
            print('*' * 80, file=logfile)
            print('EPOCH ', step, file=logfile)
            print('*' * 80, file=logfile)
            loss = model(sent, actions, corpus.dictionary, verbose=True, file=logfile)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()

            print('Step {} | loss {:.3f}'.format(step, loss.data[0]))

    except KeyboardInterrupt:
        print('Exiting training early.')

    print('*' * 80, file=logfile)
    print('PARSING ', file=logfile)
    print('*' * 80, file=logfile)

    model.eval()
    parser = model.parse(sent, corpus.dictionary, file=logfile)
    torch.save(model, CHECKFILE)

    print('Finished parsing.', file=logfile)
    print(parser, file=logfile)

if args.mode == 'parse':
    sent, actions = next(batches)
    parser = model.parse(sent, corpus.dictionary, file=logfile)
