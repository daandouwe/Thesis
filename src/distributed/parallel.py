#!/usr/bin/env python
import os
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    model = MLP(10, 10, 10)
    model.cuda()
    input = Variable(torch.zeros(17, 10).uniform_()).cuda()
    print(input)

    print('GPUs availlable: {}'.format(torch.cuda.device_count()))

    devices = [0, 1, 2, 3]
    replicas = nn.parallel.replicate(model, devices=devices)
    # input = [[input] for i in range(len(devices))]
    inputs = nn.parallel.scatter(input, devices)
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    print(outputs)
