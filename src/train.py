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

######################################################################
# Parse command line arguments.
######################################################################
parser = argparse.ArgumentParser(description='Discriminative RNNG parser')
parser.add_argument('--data', type=str, default='../tmp/ptb',
                    help='location of the data corpus')
parser.add_argument('--emb_dim', type=int, default=20,
                    help='size of word embeddings')
parser.add_argument('--lstm_hidden', type=int, default=20,
                    help='number of hidden units in LSTM')
parser.add_argument('--lstm_num_layers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--mlp_hidden', type=int, default=100,
                    help='number of hidden units in arc MLP')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=5.,
                    help='clipping gradient norm at this value')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.33,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--log_every', type=int, default=10,
                    help='report interval')
parser.add_argument('--plot_every', type=int, default=100,
                    help='plot interval')
parser.add_argument('--disable_cuda', action='store_true',
                    help='disable CUDA')
args = parser.parse_args()

# Create folders for logging and checkpoints
SUBDIR = get_subdir_string(args)
LOGDIR = os.path.join('log', SUBDIR)
CHECKDIR = os.path.join('checkpoints', SUBDIR)
LOGFILE = os.path.join(LOGDIR, 'train.log')
CHECKFILE = os.path.join(CHECKDIR, 'model.pt')
os.mkdir(LOGDIR)
os.mkdir(CHECKDIR)

# Create a logger
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG, filename=LOGFILE,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Set seed for reproducibility
torch.manual_seed(args.seed)

# Set cuda
args.cuda = not args.disable_cuda and torch.cuda.is_available()
######################################################################
# Initialize model.
######################################################################
corpus = Corpus(data_path=args.data)
model = RNNG(
    vocab_size=len(corpus.dictionary.w2i),
    stack_size=len(corpus.dictionary.s2i),
    action_size=len(corpus.dictionary.a2i),
    emb_dim=args.emb_dim,
    emb_dropout=args.dropout,
    lstm_hidden=args.lstm_hidden,
    lstm_num_layers=args.lstm_num_layers,
    lstm_dropout=args.dropout,
    mlp_hidden=args.mlp_hidden,
    cuda=args.cuda
)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    model.cuda()

logger.info(
        'VOCAB | words {} | stack {} | actions {}'.format(
    len(corpus.dictionary.w2i),
    len(corpus.dictionary.s2i),
    len(corpus.dictionary.a2i))
)

######################################################################
# The training step.
######################################################################
def train():
    """
    Performs one epoch of training.
    """
    for step, batch in enumerate(batches, 1):
        stack, buffer, history, action = batch
        out = model(stack, buffer, history)
        loss = criterion(out, action)

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        # Logging info
        if step % args.log_every == 0:
            losses.append(loss.cpu().data.numpy()[0])
            logger.info(
                '| Step {}/{} | Avg loss {:.4f} | {:.0f} sents/sec |'.format(
            step, corpus.train.num_batches,
            np.mean(losses[-args.log_every:]),
            args.batch_size*args.log_every / timer.elapsed()
                )
            )

            print(
                '| Step {}/{} | Avg loss {:.4f} | {:.0f} sents/sec |'.format(
            step, corpus.train.num_batches,
            np.mean(losses[-args.log_every:]),
            args.batch_size*args.log_every / timer.elapsed()
                )
            )

batches = corpus.train.batches(args.batch_size, length_ordered=True, cuda=args.cuda)
timer = Timer()
losses = []
train()
