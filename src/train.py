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
parser.add_argument('--data', type=str, default='../tmp/ptb',
                    help='location of the data corpus')
parser.add_argument('--outdir', type=str, default='',
                    help='location to make output log and checkpoint folders')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate')
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
OUTDIR = os.path.join(args.outdir, 'out', SUBDIR)
os.mkdir(LOGDIR)
os.mkdir(CHECKDIR)
os.mkdir(OUTDIR)
LOGFILE = os.path.join(LOGDIR, 'train.log')
CHECKFILE = os.path.join(CHECKDIR, 'model.pt')
# OUTFILE = os.path.join(OUTDIR, 'train.predict.txt')
OUTFILE = 'out/train.predict.txt'

corpus = Corpus(data_path=args.data, textline='lower')
batches = corpus.train.batches(length_ordered=False, shuffle=False)
num_batches = len(batches)

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

losses = []
timer = Timer()
# sent, indices, actions = next(batches)
try:
    for step, batch in enumerate(batches, 1):
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

except KeyboardInterrupt:
    print('Exiting training early.')

def predict(verbose=False, max_lines=None):
    model.eval()
    sentences = get_sentences(args.data + '.oracle')
    batches = corpus.train.batches(length_ordered=False, shuffle=False)
    for i, batch in enumerate(batches):
        if verbose:
            print(i, end='\r')
        if i is not None:
            if i >= max_lines:
                break
        sent, indices, actions = batch
        parser = model.parse(sent, indices)
        sentences[i]['actions'] = parser.actions
    return sentences[:i]

def write_pred(sentences, outfile, verbose=False):
    with open(outfile, 'w') as f:
        for i, sent_dict in enumerate(sentences):
            if verbose: print(i, end='\r')
            print(sent_dict['tree'], file=f)
            print(sent_dict['tags'], file=f)
            print(sent_dict['upper'], file=f)
            print(sent_dict['lower'], file=f)
            print(sent_dict['unked'], file=f)
            print('\n'.join(sent_dict['actions']), file=f)
            print(file=f)

print('Predicting.')
pred_sentences = predict(verbose=True, max_lines=10)
write_pred(pred_sentences, OUTFILE)
print('Finished')

torch.save(model, CHECKFILE)
