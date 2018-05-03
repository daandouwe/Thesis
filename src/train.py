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
from utils import Timer

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M')
logger = logging.getLogger(__name__)

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
parser.add_argument('--mlp_hidden', type=int, default=50,
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
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--print_every', type=int, default=10,
                    help='report interval')
parser.add_argument('--plot_every', type=int, default=100,
                    help='plot interval')
parser.add_argument('--save', type=str,  default='models/model.pt',
                    help='path to save the final model')
args = parser.parse_args()

torch.manual_seed(args.seed)

######################################################################
# Useful functions.
######################################################################
corpus = Corpus(data_path=args.data)
model = RNNG(vocab_size=len(corpus.dictionary.w2i),
                stack_size=len(corpus.dictionary.s2i),
                action_size=len(corpus.dictionary.a2i),
                emb_dim=args.emb_dim,
                emb_dropout=args.dropout,
                lstm_hidden=args.lstm_hidden,
                lstm_num_layers=args.lstm_num_layers,
                lstm_dropout=args.dropout,
                mlp_hidden=args.mlp_hidden)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

######################################################################
# The training step.
######################################################################
def train():
    """
    Performs one epoch of training.
    """
    for step, batch in enumerate(batches, 1):
        stack, buffer, history, action = next(batches)
        out = model(stack, buffer, history)
        loss = criterion(out, action)

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        if step % args.print_every == 0:
            losses.append(loss.data.numpy()[0])
            logger.info(
                '| Step {} | Avg loss {:.4f} | {:.0f} sents/sec |'.format(
            step, np.mean(losses[-args.print_every:]),
            args.batch_size*args.print_every/timer.elapsed()))

batches = corpus.train.batches(args.batch_size, length_ordered=True)
timer = Timer()
losses = []

train()



# ######################################################################
# # Train!
# ######################################################################
# print('Start of training..')
# timer = Timer()
# n_batches = len(corpus.train.words) // args.batch_size
# train_loss, train_acc, val_acc, test_acc = [], [], [], []
# best_val_acc, best_epoch = 0, 0
# fig, ax = plt.subplots()
# try:
#     for epoch in range(1, args.epochs+1):
#         epoch_start_time = time.time()
#         # Turn on dropout.
#         model.train()
#         # Get a new set of shuffled training batches.
#         train_batches = corpus.train.batches(args.batch_size, length_ordered=False)
#         for step, batch in enumerate(train_batches, 1):
#             words, tags, heads, labels = batch
#             S_arc, S_lab, loss = train(batch)
#             train_loss.append(loss)
#             if step % args.print_every == 0:
#                 arc_train_acc = arc_accuracy(S_arc, heads)
#                 lab_train_acc = lab_accuracy(S_lab, heads, labels)
#                 train_acc.append([arc_train_acc, lab_train_acc])
#                 print('Epoch {} | Step {}/{} | Avg loss {:.4f} | Arc accuracy {:.2f}% | '
#                       'Label accuracy {:.2f}% | {:.0f} sents/sec |'
#                         ''.format(epoch, step, n_batches, np.mean(train_loss[-args.print_every:]),
#                         100*arc_train_acc, 100*lab_train_acc,
#                         args.batch_size*args.print_every/timer.elapsed()), end='\r')
#             if step % args.plot_every == 0:
#                 plot(corpus, model, fig, ax, step)
#         # Evaluate model on validation set.
#         arc_val_acc, lab_val_acc = evaluate(model, corpus)
#         val_acc.append([arc_val_acc, lab_val_acc])
#         # Save model if it is the best so far.
#         if arc_val_acc > best_val_acc:
#             save(model)
#             best_val_acc = arc_val_acc
#             best_epoch = epoch
#         write(train_loss, train_acc, val_acc)
#         # End epoch with some useful info in the terminal.
#         print('-' * 89)
#         print('| End of epoch {} | Time elapsed: {:.2f}s | Valid accuracy {:.2f}% |'
#                 ' Best accuracy {:.2f}% (epoch {})'.format(epoch,
#                 (time.time() - epoch_start_time), 100*arc_val_acc, 100*best_val_acc, best_epoch))
#         print('-' * 89)
# except KeyboardInterrupt:
#     print('-' * 89)
#     print('Exiting from training early')
