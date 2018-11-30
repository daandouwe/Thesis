from copy import deepcopy

import torch
import torch.nn as nn
import torch.distributions as dist

from data import wrap
from nn import init_lstm
from composition import (BiRecurrentComposition, AttentionComposition, LatentFactorComposition,
    COMPOSITIONS, LATENT_COMPOSITIONS)


class VarLSTMCell(nn.Module):
    """
    An LSTM cell with recurrent (variational) dropout following
    Gal and Ghahramani (2016) `A Theoretically Grounded Application of Dropout
    in Recurrent Neural Networks` (https://arxiv.org/pdf/1512.05287.pdf.)
    """
    def __init__(self, input_size, hidden_size, dropout, device=None):
        super(VarLSTMCell, self).__init__()
        assert (0 <= dropout < 1), dropout
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        self.rnn = nn.LSTMCell(input_size, hidden_size)

        self.use_dropout = (dropout > 0)  # only use dropout when actually need to
        self.keep_prob = 1.0 - dropout
        self.bernoulli = dist.Bernoulli(
            probs=torch.tensor([self.keep_prob], device=device))

        self.reset_parameters()

    def reset_parameters(self):
        init_lstm(self.rnn)

    def sample_recurrent_dropout_masks(self, batch_size):
        """Fix a new dropout mask used for recurrent dropout."""
        # Dropout mask for input.
        self.zx = self.bernoulli.sample(
            (batch_size, self.input_size)).squeeze(-1)
        # Dropout mask for hidden state.
        self.zh = self.bernoulli.sample(
            (batch_size, self.hidden_size)).squeeze(-1)

    def forward(self, x, hidden):
        """Compute the next hidden state with input x and the previous hidden state."""
        assert isinstance(hidden, tuple), hidden
        h, c = hidden
        if self.training and self.use_dropout:
            x = x.mul(self.zx).div(self.keep_prob)
            h = h.mul(self.zh).div(self.keep_prob)
        h, c = self.rnn(x, (h, c))
        return h, c


class BaseLSTM(nn.Module):
    """A multilayered LSTM with variational dropout."""
    def __init__(self, input_size, hidden_size, dropout, device=None, num_layers=2):
        super(BaseLSTM, self).__init__()
        assert (hidden_size % 2 == 0), f'hidden size must be even: {hidden_size}'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        dims = [input_size] + num_layers * [hidden_size]
        self.layers = nn.ModuleList(
            [VarLSTMCell(input, hidden, dropout=dropout, device=device)
                for input, hidden in zip(dims[:-1], dims[1:])])
        # Store hidden states of sequence internally.
        self._hidden_states = []

    def reset_hidden(self, batch_size):
        """Reset initial hidden states to zeros."""
        # Empty history.
        self._hidden_states = []
        # Add initial hidden states.
        h = torch.zeros(batch_size, self.hidden_size, device=self.device)
        self._hidden_states.append(
            [(deepcopy(h), deepcopy(h)) for _ in range(self.num_layers)])

    def initialize(self, batch_size=1):
        """Initialize for a new sequence."""
        self.reset_hidden(batch_size)
        for layer in self.layers:
            layer.sample_recurrent_dropout_masks(batch_size)

    def forward(self, x):
        """Compute the next hidden state with input x and the previous hidden state."""
        prev_states = self._hidden_states[-1]
        new_states = []
        input = x
        for i, layer in enumerate(self.layers):
            # Get previous hidden states at this layer.
            hx, cx = prev_states[i]
            # Compute new hidden states at this layer.
            hx, cx = layer(input, (hx, cx))
            # Input to next layer is new hidden state at this layer.
            input = hx
            # Accumulate the new hidden states for each layer at this timestep.
            new_states.append((hx, cx))
        # Store new hidden states of all layers.
        self._hidden_states.append(new_states)
        return hx


class StackLSTM(BaseLSTM):
    """LSTM used to encode the stack of a transition based parser."""
    def __init__(self, input_size, hidden_size, dropout, device=None, composition='basic', num_factors=10):
        super(StackLSTM, self).__init__(input_size, hidden_size, dropout, device)
        # Composition function.
        assert composition in COMPOSITIONS, composition
        self.composition_type = composition
        self.requires_kl = (composition in LATENT_COMPOSITIONS)
        if composition == 'basic':
            self.composition = BiRecurrentComposition(input_size, 2, dropout, device=device)
        elif composition == 'attention':
            self.composition = AttentionComposition(input_size, 2, dropout, device=device)
        elif composition == 'latent-factors':
            self.composition = LatentFactorComposition(num_factors, input_size, 2, dropout, device=device)

    def _reset_hidden(self, sequence_len):
        """Reset the hidden state to right before the sequence was opened."""
        del self._hidden_states[-sequence_len:]


class HistoryLSTM(BaseLSTM):
    """LSTM used to encode the history of actions of a transition based parser."""
    def __init__(self, *args, **kwargs):
        super(HistoryLSTM, self).__init__(*args, **kwargs)


class TerminalLSTM(BaseLSTM):
    """LSTM used to encode the generated word of a generative transition based parser."""
    def __init__(self, *args, **kwargs):
        super(TerminalLSTM, self).__init__(*args, **kwargs)


class BufferLSTM(nn.Module):
    """A straightforward LSTM but wrapped to hide internals such as selection of output."""
    def __init__(self, input_size, hidden_size, num_layers, dropout, device=None):
        super(BufferLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, dropout=dropout, num_layers=num_layers,
            batch_first=True, bidirectional=False)

    def forward(self, x):
        """Encode and return the output hidden states."""
        h, _ = self.rnn(x)
        return h
