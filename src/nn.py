import copy

import torch
import torch.nn as nn
from torch.autograd import Variable

from data import wrap

class MLP(nn.Module):
    """A simple multilayer perceptron with one hidden layer and dropout."""
    def __init__(self, input_size, hidden_size, output_size, dropout=0., activation='Tanh'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act_fn = getattr(nn, activation)()
        self.dropout = nn.Dropout(p=dropout)
        self.initialize()

    def initialize(self):
        """Initialize parameters with Glorot."""
        for param in self.parameters():
            if param.dim() > 1 and param.requires_grad:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act_fn(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    mlp = MLP(2, 3, 3)
