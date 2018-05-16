import torch
import torch.nn as nn
from torch.autograd import Variable

class StackLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cuda=False):
        super(StackLSTM, self).__init__()
        self.cuda = cuda
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.initialize_hidden()

        self.rnn = nn.LSTMCell(input_size, hidden_size)

        self.stack
        self.hidden_states = []

    def initialize_hidden(self, batch_size=1):
        """Returns empty initial hidden state for each cell."""
        hx = Variable(torch.zeros(batch_size, self.hidden_size))
        cx = Variable(torch.zeros(batch_size, self.hidden_size))
        if self.cuda:
            hx = hx.cuda()
            cx = cx.cuda()
        self.hx, self.cx = hx, cx

    def forward(self, x):
        # input (batch, input_size)
        self.hx, self.cx = self.rnn(x, (self.hx, self.cx))
        self.hidden_states.append((self.hx, self.cx)) # add cell states to memory
        return self.hx
