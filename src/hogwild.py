#!/usr/bin/env python
import os

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.autograd import Variable

from data import Corpus
from model import RNNG
from utils import Timer, get_subdir_string

def train(model, rank, batch_chunk):
    start = rank*batch_chunk
    stop = (rank+1)*batch_chunk
    timer = Timer()
    for i, batch in enumerate(train_batches[start:stop]):
        sent, indices, actions = batch
        loss = model(sent, indices, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 1:
            print('p: {}, step {}, loss: {}'.format(rank, i, loss.data[0]))
            if rank == 0:
                sent_per_sec = 10 / timer.elapsed()
                total_speed = num_processes * sent_per_sec
                print('Speed: {:.3f}sents/sec'.format(total_speed))

###################
# Mutliprocessing #
###################

if __name__ == '__main__':
    # Initialize model and data.
    corpus = Corpus(data_path='../tmp', textline='unked')
    train_batches =  corpus.train.batches(length_ordered=False, shuffle=False)
    num_batches = len(train_batches)
    model = RNNG(dictionary=corpus.dictionary,
                 emb_dim=100,
                 emb_dropout=0.3,
                 lstm_hidden=100,
                 lstm_num_layers=1,
                 lstm_dropout=0.3,
                 mlp_hidden=500,
                 use_cuda=False)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3)

    # Initialize parallel stuff.
    num_processes = 3
    batch_chunk = num_batches // num_processes

    # NOTE: this is required for the ``fork`` method to work
    model.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(model, rank, batch_chunk))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
