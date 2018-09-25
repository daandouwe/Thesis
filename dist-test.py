#!/usr/bin/env python

"""run.py:"""
import argparse
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.distributed as dist
import torch.nn.functional as F
from torch.multiprocessing import Process


def run(rank, size):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    # Restrict the number of threads that each process uses.
    torch.set_num_threads(1)

    # Random objective.
    data, target = Variable(torch.ones(1, 500).uniform_()), Variable(torch.ones(1, 500).uniform_())
    # Dummy model.
    model = nn.Linear(500, 500)  # y = Wx + b

    optimizer = torch.optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)

    for epoch in range(100000):
        output = model(data)
        loss = torch.mean((output - target)**2)

        optimizer.zero_grad()
        loss.backward()
        average_gradients(model)
        optimizer.step()
        if rank == 0 and epoch % 100 == 0:
            print(f'Epoch {epoch:,}: {loss.data[0]:.2e}')


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    # dist.init_process_group(backend, init_method='env://')
    fn(rank, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--size", type=int, default=8)
    args = parser.parse_args()

    print('Number of processes:', args.size)

    processes = []
    for rank in range(args.size):
        p = Process(target=init_processes, args=(rank, args.size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
