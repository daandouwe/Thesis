import copy

import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.init as init

from data import wrap


def orthogonal_init(lstm):
    for name, param in lstm.named_parameters():
        if name.startswith('weight'):
            init.orthogonal_(param)


def bias_init(lstm):
    """Positive forget gate bias (Jozefowicz et al., 2015)."""
    for name, param in lstm.named_parameters():
        if name.startswith('bias'):
            init.constant_(param, 0.)
            dim = param.size(0)
            param[dim//4:dim//2].data.fill_(1.)


def init_lstm(lstm, orthogonal=True):
    """Initialize the forget bias and weights of LSTM."""
    bias_init(lstm)
    if orthogonal:
        orthogonal_init(lstm)


class BiRecurrentEncoder(nn.Module):
    """A bidirectional RNN encoder for unpadded batches."""
    def __init__(self, input_size, hidden_size, num_layers, dropout, batch_first=True, device=None):
        super(BiRecurrentEncoder, self).__init__()
        assert hidden_size % 2 == 0, 'hidden size must be even'
        self.device = device
        self.fwd_rnn = nn.LSTM(input_size, hidden_size//2, num_layers,
                               batch_first=batch_first, dropout=dropout)
        self.bwd_rnn = nn.LSTM(input_size, hidden_size//2, num_layers,
                               batch_first=batch_first, dropout=dropout)
        init_lstm(self.fwd_rnn)
        init_lstm(self.bwd_rnn)
        self.to(device)

    def _reverse(self, tensor):
        idx = [i for i in range(tensor.size(1) - 1, -1, -1)]
        idx = wrap(idx, device=self.device)
        return tensor.index_select(1, idx)

    def forward(self, head, children):
        """Forward pass that works for unpadded, i.e. equal length, batches."""
        # children shape (batch, seq, dim)
        # head shape (batch, dim)
        xf = torch.cat((head.unsqueeze(1), children), dim=1)  # [NP, the, black, cat]
        xb = torch.cat((head.unsqueeze(1), self._reverse(children)), dim=1)  # [NP, cat, black, the]

        hf, _ = self.fwd_rnn(xf)  # (batch, seq, hidden_size//2)
        hb, _ = self.bwd_rnn(xb)  # (batch, seq, hidden_size//2)

        # Select final representation.
        hf = hf[:, -1, :]  # (batch, hidden_size//2)
        hb = hb[:, -1, :]  # (batch, hidden_size//2)

        h = torch.cat((hf, hb), dim=-1)  # (batch, hidden_size)
        return h


class AttentionEncoder(nn.Module):
    """A bidirectional RNN encoder for unpadded batches."""
    def __init__(self, input_size, hidden_size, num_layers, dropout, batch_first=True, device=None):
        super(AttentionEncoder, self).__init__()
        assert hidden_size % 2 == 0, 'hidden size must be even'
        self.device = device
        self.fwd_rnn = nn.LSTM(input_size, hidden_size//2, num_layers,
                               batch_first=batch_first, dropout=dropout)
        self.bwd_rnn = nn.LSTM(input_size, hidden_size//2, num_layers,
                               batch_first=batch_first, dropout=dropout)
        self.V = nn.Parameter(torch.ones((hidden_size, hidden_size), device=device, dtype=torch.float))
        self.linear = nn.Linear(2*hidden_size, hidden_size)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        init_lstm(self.fwd_rnn)
        init_lstm(self.bwd_rnn)
        nn.init.xavier_uniform_(self.V)
        self.to(device)

    def _reverse(self, tensor):
        idx = [i for i in range(tensor.size(1) - 1, -1, -1)]
        idx = wrap(idx, device=self.device)
        return tensor.index_select(1, idx)

    def forward(self, head, children):
        """Forward pass that works for unpadded, i.e. equal length, batches."""
        # children shape (batch, seq, dim)
        # head shape (batch, dim)
        xf = torch.cat((head.unsqueeze(1), children), dim=1)  # [NP, the, black, cat]
        xb = torch.cat((head.unsqueeze(1), self._reverse(children)), dim=1)  # [NP, black, cat, the]

        hf, _ = self.fwd_rnn(xf)  # (batch, seq, hidden_size//2)
        hb, _ = self.bwd_rnn(xb)  # (batch, seq, hidden_size//2)

        c = torch.cat((hf, hb), dim=-1)  # (batch, seq, hidden_size)
        a = c @ self.V @ head.transpose(0, 1)  # (batch, seq, 1)
        a = a.squeeze(-1)  # (batch, seq)
        a = self.softmax(a)  # (batch, seq)
        m = a @ c  # (batch, seq) @ (batch, seq, hidden_size) = (batch, 1, hidden_size)
        m = m.squeeze(1)  # (batch, hidden_size)

        # Optional: store computed attention internally for retreival.
        self.attn = a

        x = torch.cat((head, m), dim=-1)  # (batch, 2*hidden_size)
        g = self.sigmoid(self.linear(x))  # (batch, hidden_size)
        c = g * head + (1 - g) * m  # (batch, hidden_size)
        return c


class BaseLSTM(nn.Module):
    """A simple two-layered LSTM inherited by StackLSTM and HistoryLSTM."""
    def __init__(self, input_size, hidden_size, dropout, device=None):
        super(BaseLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size  # Must be even number, see composition function.
        self.device = device  # GPU or CPU

        self.rnn_1 = nn.LSTMCell(input_size, hidden_size)
        self.rnn_2 = nn.LSTMCell(hidden_size, hidden_size)

        # Were we store all intermediate computed hidden states.
        # Last item in _hidden_states_2 is used as the representation.
        self._hidden_states_1 = []  # layer 1
        self._hidden_states_2 = []  # layer 2

        # Used for custom dropout.
        self.keep_prob = 1.0 - dropout
        self.bernoulli = dist.Bernoulli(
            probs=torch.tensor([self.keep_prob], device=device)
        )
        init_lstm(self.rnn_1)
        init_lstm(self.rnn_2)
        self.initialize_hidden()
        self.to(device)

    def sample_recurrent_dropout_mask(self, batch_size):
        """Fix a new dropout mask used for recurrent dropout."""
        self._dropout_mask = self.bernoulli.sample(
            (batch_size, self.hidden_size)
        ).squeeze(-1)

    def dropout(self, x):
        """Custom recurrent dropout: same mask for the whole sequence."""
        scale = 1 / self.keep_prob  # Scale the weights up to compensate for dropping out.
        return x * self._dropout_mask * scale

    def initialize_hidden(self, batch_size=1):
        """Set initial hidden state to zeros."""
        c = copy.deepcopy
        self._hidden_states_1 = []
        self._hidden_states_2 = []
        h0 = torch.zeros(batch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros(batch_size, self.hidden_size, device=self.device)
        self.hx1, self.cx1 = h0, c0
        self.hx2, self.cx2 = c(h0), c(c0)
        self._hidden_states_1.append((self.hx1, self.cx1))
        self._hidden_states_2.append((self.hx2, self.cx2))
        self.sample_recurrent_dropout_mask(batch_size)

    def forward(self, x):
        """Compute the next hidden state with input x and the previous hidden state.

        Args:
            x (tensor): shape (batch, input_size).
        """
        # First layer
        self.hx1, self.cx1 = self.rnn_1(x, (self.hx1, self.cx1))
        if self.training:
            self.hx1, self.cx1 = self.dropout(self.hx1), self.dropout(self.cx1)
        # Second layer
        self.hx2, self.cx2 = self.rnn_2(self.hx1, (self.hx2, self.cx2))
        if self.training:
            self.hx2, self.cx2 = self.dropout(self.hx2), self.dropout(self.cx2)
        # Add cell states to memory.
        self._hidden_states_1.append((self.hx1, self.cx1))
        self._hidden_states_2.append((self.hx2, self.cx2))
        # Return hidden state of second layer
        return self.hx2


class StackLSTM(BaseLSTM):
    """A Stack-LSTM used to encode the stack of a transition based parser."""
    def __init__(self, input_size, hidden_size, dropout, device=None, attn_comp=False):
        super(StackLSTM, self).__init__(input_size, hidden_size, dropout, device)
        # Composition function.
        if attn_comp:
            self.composition = AttentionEncoder(input_size, input_size, 2, dropout, device=device)
        else:
            self.composition = BiRecurrentEncoder(input_size, input_size, 2, dropout, device=device)

    def _reset_hidden(self, sequence_len):
        """Reset the hidden state to before opening the sequence."""
        del self._hidden_states_1[-sequence_len:], self._hidden_states_2[-sequence_len:]
        # self._hidden_states_1 = self._hidden_states_1[:-sequence_len]
        # self._hidden_states_2 = self._hidden_states_2[:-sequence_len]
        self.hx1, self.cx1 = self._hidden_states_1[-1]
        self.hx2, self.cx2 = self._hidden_states_2[-1]

    def reduce(self, children, head):
        """Reduce a nonterminal sequence.

        Computes a BiRNN represesentation for the sequence, then replaces
        the reduced sequence of hidden states with this one representation.
        """
        # Length of sequence (length of children plus one head node).
        length = children.size(1) + 1
        # Move hidden state back to before we opened the nonterminal.
        self._reset_hidden(length)
        # Return computed composition.
        return self.composition(head, children)


class HistoryLSTM(BaseLSTM):
    """An LSTM used to encode the history of actions of a transition based parser."""
    def __init__(self, input_size, hidden_size, dropout, device=None):
        super(HistoryLSTM, self).__init__(
            input_size,
            hidden_size,
            dropout,
            device
        )


class TerminalLSTM(BaseLSTM):
    """An LSTM used to encode the history of actions of a transition based parser."""
    def __init__(self, input_size, hidden_size, dropout, device=None):
        super(TerminalLSTM, self).__init__(
            input_size,
            hidden_size,
            dropout,
            device
        )


class BufferLSTM(nn.Module):
    """A straightforward lstm but wrapped to hide internals such as selection of output."""
    def __init__(self, input_size, hidden_size, num_layers, dropout, device=None):
        super(BufferLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, dropout=dropout, num_layers=num_layers,
                           batch_first=True, bidirectional=False)

    def forward(self, x):
        """Encode and return the output hidden states."""
        h, _ = self.rnn(x)
        return h


if __name__ == '__main__':
    history_encoder = HistoryLSTM(2, 3, 0.1)
    # init_lstm(history_encoder.rnn_1)

    for name, param in history_encoder.rnn_1.named_parameters():
        print(name)

    buffer_encoder = BufferLSTM(2, 3, 2, 0.1)
    init_lstm(buffer_encoder.rnn)
    for name, param in buffer_encoder.rnn.named_parameters():
        print(name)

    orthogonal_init(history_encoder.rnn_1)
