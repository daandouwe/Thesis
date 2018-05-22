import argparse
import logging
import os
import csv
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from newdata import Corpus
from newmodel import RNNG
from utils import Timer, get_subdir_string

LR = 1e-3
CLIP = 5.

torch.manual_seed(42)

corpus = Corpus(data_path="../tmp/ptb")
batches = corpus.train.batches(length_ordered=False, shuffle=False)

model = RNNG(vocab_size=len(corpus.dictionary.w2i),
             stack_size=len(corpus.dictionary.s2i),
             action_size=len(corpus.dictionary.a2i),
             emb_dim=20, emb_dropout=0.3,
             lstm_hidden=20, lstm_num_layers=1, lstm_dropout=0.3,
             mlp_hidden=50, cuda=False)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for step in range(100):
    sent, actions = next(batches)

    loss = model(sent, actions, corpus.dictionary, verbose=False)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), CLIP)
    optimizer.step()

    print('Step {} | loss {:.3f}'.format(step, loss.data.numpy()[0]))
