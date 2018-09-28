from copy import deepcopy

import torch
import torch.nn as nn
import torch.distributions as dist

from data import wrap
from nn import init_lstm
from composition import BiRecurrentComposition, AttentionComposition, LatentFactorComposition


COMPOSITIONS = ('basic', 'attention', 'latent-factors', 'latent-attention')

LATENT_COMPOSITIONS = ('latent-factors', 'latent-attention')


class BaseLSTM(nn.Module):
    """A two-layered LSTM."""
    def __init__(self, input_size, hidden_size, dropout, device=None):
        super(BaseLSTM, self).__init__()
        assert (hidden_size % 2 == 0), f'hidden size must be even: {hidden_size}'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        self.rnn1 = nn.LSTMCell(input_size, hidden_size)
        self.rnn2 = nn.LSTMCell(hidden_size, hidden_size)

        # Were we store all intermediate computed hidden states.
        # Last item in _hidden_states_layer2 is used as the representation.
        self._hidden_states_layer1 = []
        self._hidden_states_layer2 = []

        # Used for custom dropout.
        self.keep_prob = 1.0 - dropout
        self.bernoulli = dist.Bernoulli(
            probs=torch.tensor([self.keep_prob], device=device))

        # Initialize all layers.
        init_lstm(self.rnn1)
        init_lstm(self.rnn2)
        self.initialize_hidden()
        self.to(device)

    def sample_recurrent_dropout_mask(self, batch_size):
        """Fix a new dropout mask used for recurrent dropout."""
        self._dropout_mask = self.bernoulli.sample(
            (batch_size, self.hidden_size)).squeeze(-1)

    def dropout(self, x):
        """Custom recurrent dropout: same mask for the whole sequence."""
        scale = 1 / self.keep_prob  # Scale the weights up to compensate for dropping out.
        return x * self._dropout_mask * scale

    def initialize_hidden(self, batch_size=1):
        """Set initial hidden state to zeros."""
        self._hidden_states_layer1 = []
        self._hidden_states_layer2 = []
        # TODO: make first hidden states trainable.
        h0 = torch.zeros(batch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros(batch_size, self.hidden_size, device=self.device)
        self.hx1, self.cx1 = h0, c0
        self.hx2, self.cx2 = deepcopy(h0), deepcopy(c0)
        self._hidden_states_layer1.append((self.hx1, self.cx1))
        self._hidden_states_layer2.append((self.hx2, self.cx2))
        self.sample_recurrent_dropout_mask(batch_size)

    def forward(self, x):
        """Compute the next hidden state with input x and the previous hidden state."""
        # First layer
        hx1, cx1 = self.rnn1(x, (self.hx1, self.cx1))
        if self.training:
            hx1, cx1 = self.dropout(hx1), self.dropout(cx1)
        # Second layer
        hx2, cx2 = self.rnn2(hx1, (self.hx2, self.cx2))
        if self.training:
            hx2, cx2 = self.dropout(hx2), self.dropout(cx2)
        # Add cell states to history.
        self._hidden_states_layer1.append((hx1, cx1))
        self._hidden_states_layer2.append((hx2, cx2))
        # Set new current hidden states
        self.hx1, self.cx1 = hx1, cx1
        self.hx2, self.cx2 = hx2, cx2
        # Return hidden state of second layer
        return hx2


class StackLSTM(BaseLSTM):
    """LSTM used to encode the stack of a transition based parser."""
    def __init__(self, input_size, hidden_size, dropout, device=None, composition='basic'):
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
            self.composition = LatentFactorComposition(10, input_size, 2, dropout, device=device)

    def _reset_hidden(self, sequence_len):
        """Reset the hidden state to right before the sequence was opened."""
        del self._hidden_states_layer1[-sequence_len:]
        del self._hidden_states_layer2[-sequence_len:]
        self.hx1, self.cx1 = self._hidden_states_layer1[-1]
        self.hx2, self.cx2 = self._hidden_states_layer2[-1]


class HistoryLSTM(BaseLSTM):
    """LSTM used to encode the history of actions of a transition based parser."""
    def __init__(self, *args):
        super(HistoryLSTM, self).__init__(*args)


class TerminalLSTM(BaseLSTM):
    """LSTM used to encode the generated word of a generative transition based parser."""
    def __init__(self, *args):
        super(TerminalLSTM, self).__init__(*args)


class BufferLSTM(nn.Module):
    """A straightforward lstm but wrapped to hide internals such as selection of output."""
    def __init__(self, input_size, hidden_size, num_layers, dropout, device=None):
        super(BufferLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, dropout=dropout, num_layers=num_layers,
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
