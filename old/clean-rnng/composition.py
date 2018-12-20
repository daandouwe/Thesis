import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.init as init

# from data import wrap
# from concrete import BinaryConcrete, Concrete
# from loss import AnnealTemperature

from nn import init_lstm


COMPOSITIONS = ('basic', 'attention', 'latent-factors', 'latent-attention')


LATENT_COMPOSITIONS = ('latent-factors', 'latent-attention')


class BiRecurrentComposition(nn.Module):
    """Bidirectional RNN composition function."""
    def __init__(self, input_size, num_layers, dropout, batch_first=True, device=None):
        super(BiRecurrentComposition, self).__init__()
        assert input_size % 2 == 0, 'hidden size must be even'
        self.device = device
        self.fwd_rnn = nn.LSTM(
            input_size, input_size//2, num_layers, batch_first=batch_first, dropout=dropout)
        self.bwd_rnn = nn.LSTM(
            input_size, input_size//2, num_layers, batch_first=batch_first, dropout=dropout)
        init_lstm(self.fwd_rnn)
        init_lstm(self.bwd_rnn)

    def _reverse(self, tensor):
        idx = [i for i in range(tensor.size(1) - 1, -1, -1)]
        idx = torch.tensor(idx, device=self.device)
        return tensor.index_select(1, idx)

    def forward(self, head, children):
        """Forward pass that works for unpadded, i.e. equal length, batches.
        Args:
            children (torch.tensor): shape (batch, seq, dim)
            head (torch.tensor): shape (batch, dim)
        """
        xf = torch.cat((head.unsqueeze(1), children), dim=1)  # [NP, the, black, cat]
        xb = torch.cat((head.unsqueeze(1), self._reverse(children)), dim=1)  # [NP, cat, black, the]

        hf, _ = self.fwd_rnn(xf)  # (batch, seq, input_size//2)
        hb, _ = self.bwd_rnn(xb)  # (batch, seq, input_size//2)

        # Select final representation.
        hf = hf[:, -1, :]  # (batch, input_size//2)
        hb = hb[:, -1, :]  # (batch, input_size//2)

        h = torch.cat((hf, hb), dim=-1)  # (batch, input_size)
        return h


class AttentionComposition(nn.Module):
    """Bidirectional RNN composition function with attention."""
    def __init__(self, input_size, num_layers, dropout, batch_first=True, device=None):
        super(AttentionComposition, self).__init__()
        assert input_size % 2 == 0, 'hidden size must be even'
        self.device = device
        self.fwd_rnn = nn.LSTM(
            input_size, input_size//2, num_layers, batch_first=batch_first, dropout=dropout)
        self.bwd_rnn = nn.LSTM(
            input_size, input_size//2, num_layers, batch_first=batch_first, dropout=dropout)
        self.V = nn.Parameter(
            torch.ones((input_size, input_size), device=device, dtype=torch.float))
        self.gating = nn.Linear(2*input_size, input_size)
        self.head = nn.Linear(input_size, input_size)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        init_lstm(self.fwd_rnn)
        init_lstm(self.bwd_rnn)
        nn.init.xavier_uniform_(self.V)

    def _reverse(self, tensor):
        idx = [i for i in range(tensor.size(1) - 1, -1, -1)]
        idx = torch.tensor(idx, device=self.device)
        return tensor.index_select(1, idx)

    def forward(self, head, children):
        """Forward pass that works for unpadded, i.e. equal length, batches.
        Args:
            children (torch.tensor): shape (batch, seq, dim)
            head (torch.tensor): shape (batch, dim)
        """
        print(children.requires_grad)
        print(head.requires_grad)

        xf = children  # [the, black, cat]
        xb = self._reverse(children)  # [black, cat, the]

        hf, _ = self.fwd_rnn(xf)  # (batch, seq, input_size//2)
        hb, _ = self.bwd_rnn(xb)  # (batch, seq, input_size//2)
        c = torch.cat((hf, hb), dim=-1)  # (batch, seq, input_size)

        a = c @ self.V @ head.transpose(0, 1)  # (batch, seq, 1)
        a = a.squeeze(-1)  # (batch, seq)
        a = self.softmax(a)  # (batch, seq)
        m = a @ c  # (batch, seq) @ (batch, seq, input_size) = (batch, 1, input_size)
        m = m.squeeze(1)  # (batch, input_size)

        x = torch.cat((head, m), dim=-1)  # (batch, 2*input_size)
        g = self.sigmoid(self.gating(x))  # (batch, input_size)
        t = self.head(head)  # (batch, input_size)
        c = g * t + (1 - g) * m  # (batch, input_size)
        if not self.training:
            # Store internally for inspection during prediction.
            self._attn = a.data()
            self._gate = g.data()
        return c
