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

torch.manual_seed(42)

corpus = Corpus(data_path="../tmp/ptb")
batches = corpus.train.batches(length_ordered=False, shuffle=False)

model = RNNG(vocab_size=len(corpus.dictionary.w2i),
             stack_size=len(corpus.dictionary.s2i),
             action_size=len(corpus.dictionary.a2i),
             emb_dim=20, emb_dropout=0.3,
             lstm_hidden=20, lstm_num_layers=1, lstm_dropout=0.3,
             mlp_hidden=50, cuda=False)

# sents = [1, 2, 3]
# actions = [1, 2, 3]

sent, actions = next(batches)
print(sent)

model(sent, actions, corpus.dictionary)
