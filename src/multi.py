#!/usr/bin/env python
import os

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.autograd import Variable

from data import Corpus
from model import RNNG
from utils import Timer, get_subdir_string

def forward(model, batch, queue):
    sent, indices, actions = batch
    loss = model(sent, indices, actions)
    queue.put(loss)

def parallel_forward(batches):
    # assert len(batches) == NUM_PROCESSES
    queue = mp.Queue()
    processes = []
    for rank in range(NUM_PROCESSES):
        p = mp.Process(target=forward, args=(model, batches[rank], queue))
        p.start()
        processes.append(p)
    loss = Variable(torch.zeros(1))
    for _ in processes:
        loss += queue.get()
    for p in processes:
        p.join()
    return loss

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
    NUM_PROCESSES = 8
    batch_chunk = num_batches // NUM_PROCESSES

    # NOTE: this is required for the ``fork`` method to work
    model.share_memory()
    timer = Timer()
    print_every = 10
    for i in range(len(train_batches) // 3):
        batches = train_batches[i*NUM_PROCESSES : (i+1)*NUM_PROCESSES]
        loss = parallel_forward(batches)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % print_every == 1:
            print('step {}, loss {:.3f}, {:.3f}sents/sec'.format(i, loss.data[0], print_every*NUM_PROCESSES/timer.elapsed()))
