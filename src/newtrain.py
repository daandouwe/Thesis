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


# corpus = Corpus(data_path="../tmp/ptb")
# batches = corpus.train.batches(length_ordered=True)

model = RNNG(vocab_size=10, stack_size=10, action_size=10, emb_dim=20, emb_dropout=0.3,
             lstm_hidden=20, lstm_num_layers=1, lstm_dropout=0.3, mlp_hidden=50, cuda=False)

sent = [1, 2, 3]
actions = [1, 2, 3]

model(sent, actions)
