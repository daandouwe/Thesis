import torch
import torch.nn as nn
from torch.autograd import Variable

from data import Corpus
from model import RNNG

corpus = Corpus(data_path="../tmp/ptb")
batches = corpus.train.batches(4, length_ordered=True)

model = RNNG(vocab_size=len(corpus.dictionary.w2i),
                stack_size=len(corpus.dictionary.s2i),
                action_size=len(corpus.dictionary.a2i),
                emb_dim=20,
                emb_dropout=0.3,
                lstm_hidden=50,
                lstm_num_layers=1,
                lstm_dropout=0.3,
                mlp_hidden=50)


for _ in range(3):
    stack, buffer, history, action = next(batches)
    print(stack.shape, buffer.shape, history.shape)
    out = model(stack, buffer, history)
