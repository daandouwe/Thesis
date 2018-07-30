#!/usr/bin/env python
import os
import sys

import torch
import torch.distributed as dist
from torch.multiprocessing import Process

from data import Corpus
from model import RNNG
from util import Timer, get_subdir_string

def train(rank, size, model, batches, optimizer, print_every=10):
    """Distributed Synchronous SGD Example."""
    chunk_size = len(batches) // dist.get_world_size()
    start = rank * chunk_size
    stop = (rank+1) * chunk_size
    timer = Timer()
    for i, batch in enumerate(batches[start:stop], 1):
        sent, indices, actions = batch
        loss = model(sent, indices, actions)
        optimizer.zero_grad()
        loss.backward()
        average_gradients(model)
        optimizer.step()
        if i % print_every == 0 and rank == 0:
            sents_per_sec = print_every * dist.get_world_size() / timer.elapsed()
            print('step {} | loss {:.3f} | {:.3f} sents/sec'.format(
                    i, loss.data[0], sents_per_sec))

def average_gradients(model):
    """Gradient averaging."""
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None: # some layers of mdel are not used and have no grad
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size

def init_processes(fn, *args, backend='tcp'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(*args)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Number of processes requested
        size = int(sys.argv[1])
    else:
        exit('Specify number of processors.')

    # Initialize model and data.
    corpus = Corpus(data_path='../tmp', textline='unked')
    train_batches =  corpus.train.batches(length_ordered=False, shuffle=False)
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

    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(train, rank, size, model, train_batches, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
