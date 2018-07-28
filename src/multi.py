#!/usr/bin/env python
import os
import sys

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.autograd import Variable

from data import Corpus
from model import RNNG
from util import Timer, get_subdir_string

## Method 1 ##

def forward(model, batch, output):
    sent, indices, actions = batch
    loss = model(sent, indices, actions)
    output.put(loss)

def parallel_forward(model, batches):
    # Define an output queue.
    output = mp.Queue()
    # Setup a list of processes that we want to run.
    processes = [mp.Process(target=forward, args=(model, batch, output))
                        for batch in batches]
    # Run the processes.
    for p in processes:
        p.start()
    # Exit the completed processes.
    for p in processes:
        p.join()
    # Get the results.
    losses = [output.get() for p in processes]
    return losses

## Method 2 ##

def forward_pool(model, batch):
    sent, indices, actions = batch
    loss = model(sent, indices, actions)
    return loss

def pool(batches):
    pool = mp.Pool(processes=NUM_PROCESSES)
    results = [pool.apply(forward_pool, args=(model, batch))
                        for batch in batches]
    return results

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
    NUM_PROCESSES = int(sys.argv[1])
    print('Using {} processors'.format(NUM_PROCESSES))
    batch_chunk = num_batches // NUM_PROCESSES

    # NOTE: this is required for the ``fork`` method to work
    model.share_memory()

    timer = Timer()
    print_every = 10
    for i in range(len(train_batches) // 3):
        batches = train_batches[i*NUM_PROCESSES : (i+1)*NUM_PROCESSES]
        losses = parallel_forward(model, batches)
        # losses = pool(batches) # also working
        loss = sum(losses)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # NOTE: model parameters are not updated. Why?

        if i % print_every == 1:
            print('step {}, loss {:.3f}, {:.3f}sents/sec'.format(i, loss.data[0], print_every*NUM_PROCESSES/timer.elapsed()))
