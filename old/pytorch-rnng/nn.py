import copy

import torch
import torch.nn as nn

from data import wrap


def orthogonal_init(lstm):
    for name, param in lstm.named_parameters():
        if name.startswith('weight'):
            nn.init.orthogonal_(param)


def bias_init(lstm):
    """Positive forget gate bias (Jozefowicz et al., 2015)."""
    for name, param in lstm.named_parameters():
        if name.startswith('bias'):
            nn.init.constant_(param, 0.)
            dim = param.size(0)
            param[dim//4:dim//2].data.fill_(1.)


def init_lstm(lstm, orthogonal=True):
    """Initialize the weights and forget bias of an LSTM."""
    bias_init(lstm)
    if orthogonal:
        orthogonal_init(lstm)


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
        h = self.fc1(x)
        h = self.act_fn(h)
        h = self.dropout(h)
        out = self.fc2(h)
        return out


if __name__ == '__main__':
    mlp = MLP(2, 3, 3)
