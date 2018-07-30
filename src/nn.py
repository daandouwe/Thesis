import torch
import torch.nn as nn
from torch.autograd import Variable

from data import PAD_INDEX, EMPTY_INDEX, REDUCED_INDEX, wrap

class MLP(nn.Module):
    """A simple multilayer perceptron with one hidden layer and dropout."""
    def __init__(self, input_size, hidden_size, output_size, dropout=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class BiRecurrentEncoder(nn.Module):
    """A bidirectional RNN encoder."""
    def __init__(self,input_size, hidden_size, num_layers, dropout, batch_first=True, use_cuda=False):
        super(BiRecurrentEncoder, self).__init__()
        self.forward_rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=batch_first,
                           dropout=dropout)
        self.backward_rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=batch_first,
                           dropout=dropout)
        self.use_cuda = use_cuda

    def _reverse(self, tensor):
        idx = [i for i in range(tensor.size(1) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx))
        idx = idx.cuda() if self.use_cuda else idx
        return tensor.index_select(1, idx)

    def forward(self, x):
        hf, _ = self.forward_rnn(x)                 # [batch, seq, hidden_size]
        hb, _ = self.backward_rnn(self._reverse(x)) # [batch, seq, hidden_size]

        # select final representation
        hf = hf[:, -1, :] # [batch, hidden_size]
        hb = hb[:, -1, :] # [batch, hidden_size]

        h = torch.cat((hf, hb), dim=-1) # [batch, 2*hidden_size]
        return h

class StackLSTM(nn.Module):
    """A Stack-LSTM used to encode the stack of a transition based parser."""
    def __init__(self, input_size, hidden_size, use_cuda=False):
        super(StackLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size # Must be even number, see composition function.
        self.use_cuda = use_cuda

        self.rnn = nn.LSTMCell(input_size, hidden_size)
        # BiRNN ecoder used as composition function.
        # NOTE: we use hidden_size//2 because the output of the composition function
        # is a concatenation of two hidden vectors.
        self.composition = BiRecurrentEncoder(input_size, hidden_size//2,
                                              num_layers=1, dropout=0., use_cuda=use_cuda)

        # Were we store all intermediate computed hidden states.
        # Top of this list is used as the stack embedding
        self._hidden_states = []

        self.initialize_hidden()

        if self.use_cuda:
            self.cuda()

    def _reset_hidden(self, sequence_len):
        """Reset the hidden state to before opening the sequence."""
        self._hidden_states = self._hidden_states[:-sequence_len]
        self.hx, self.cx = self._hidden_states[-1]

    def initialize_hidden(self, batch_size=1):
        """Set initial hidden state to zeros."""
        self._hidden_states = []
        hx = Variable(torch.zeros(batch_size, self.hidden_size))
        cx = Variable(torch.zeros(batch_size, self.hidden_size))
        if self.use_cuda:
            hx = hx.cuda()
            cx = cx.cuda()
        self.hx, self.cx = hx, cx

    def reduce(self, sequence):
        """Reduce a nonterminal sequence.

        Computes a BiRNN represesentation for the sequence, then replaces
        the reduced sequence of hidden states with this one representation.
        """
        length = sequence.size(1) - 1 # length of sequence (minus extra nonterminal at end)
        # Move hidden state back to before we opened the nonterminal.
        self._reset_hidden(length)
        return self.composition(sequence)

    def forward(self, x):
        """Compute the next hidden state with input x and the previous hidden state.

        Args: x is shape (batch, input_size).
        """
        self.hx, self.cx = self.rnn(x, (self.hx, self.cx))
        # Add cell states to memory.
        self._hidden_states.append((self.hx, self.cx))
        return self.hx


class HistoryLSTM(nn.Module):
    """A LSTM used to encode the history of actions of a transition based parser."""
    def __init__(self, input_size, hidden_size, use_cuda=False):
        super(HistoryLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size # Must be even number, see composition function.
        self.use_cuda = use_cuda

        self.rnn = nn.LSTMCell(input_size, hidden_size)

        # Were we store all intermediate computed hidden states.
        # Top of this list is used as the stack embedding
        self._hidden_states = []

        self.initialize_hidden()

        if self.use_cuda:
            self.cuda()

    def initialize_hidden(self, batch_size=1):
        """Set initial hidden state to zeros."""
        self._hidden_states = []
        hx = Variable(torch.zeros(batch_size, self.hidden_size))
        cx = Variable(torch.zeros(batch_size, self.hidden_size))
        if self.use_cuda:
            hx = hx.cuda()
            cx = cx.cuda()
        self.hx, self.cx = hx, cx

    def forward(self, x):
        """Compute the next hidden state with input x and the previous hidden state.

        Args: x is shape (batch, input_size).
        """
        self.hx, self.cx = self.rnn(x, (self.hx, self.cx))
        # Add cell states to memory.
        self._hidden_states.append((self.hx, self.cx))
        return self.hx

class RecurrentCharEmbedding(nn.Module):
    """Simple model for character-level word embeddings

    Source: https://github.com/bastings/parser/blob/extended_parser/parser/nn.py
    """
    def __init__(self, nchars, emb_dim, hidden_size, output_dim, dropout=0.33, bi=True):
        super(RecurrentCharEmbedding, self).__init__()

        self.embedding = nn.Embedding(nchars, emb_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, num_layers=1,
                            batch_first=True, dropout=dropout, bidirectional=bi)

        rnn_dim = hidden_size * 2 if bi else hidden_size
        self.linear = nn.Linear(rnn_dim, output_dim, bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        cuda = torch.cuda.is_available()

        # For single token words
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # Sort x by length.
        lengths = (x != PAD_INDEX).long().sum(-1)
        sorted_lengths, sort_idx = torch.sort(lengths, descending=True)
        sort_idx = sort_idx.cuda() if cuda else sort_idx
        x = x.index_select(0, sort_idx)

        # embed chars
        x = self.embedding(x)
        x = self.dropout(x)
        sorted_lengths = [i for i in sorted_lengths.data]
        x = nn.utils.rnn.pack_padded_sequence(x, sorted_lengths, batch_first=True)

        out, _ = self.rnn(x)

        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        sorted_lengths = wrap(sorted_lengths) - 1
        sorted_lengths = sorted_lengths.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out.size(-1))
        sorted_lengths = sorted_lengths.cuda() if cuda else sorted_lengths
        out = torch.gather(out, 1, sorted_lengths).squeeze(1) # keep only final embedding

        out = self.relu(out)
        out = self.linear(out)

        # PUT BACK INTO ORIGINAL ORDER!!!

        return out
