import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.init as init

from data import wrap
from concrete import BinaryConcrete, Concrete
from loss import AnnealTemperature

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
        self.to(device)

    def _reverse(self, tensor):
        idx = [i for i in range(tensor.size(1) - 1, -1, -1)]
        idx = wrap(idx, device=self.device)
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
        self.to(device)

    def _reverse(self, tensor):
        idx = [i for i in range(tensor.size(1) - 1, -1, -1)]
        idx = wrap(idx, device=self.device)
        return tensor.index_select(1, idx)

    def forward(self, head, children):
        """Forward pass that works for unpadded, i.e. equal length, batches.
        Args:
            children (torch.tensor): shape (batch, seq, dim)
            head (torch.tensor): shape (batch, dim)
        """
        xf = children  # [the, black, cat]
        xb = self._reverse(children)  # [black, cat, the]

        hf, _ = self.fwd_rnn(xf)  # (batch, seq, input_size//2)
        hb, _ = self.bwd_rnn(xb)  # (batch, seq, input_size//2)
        c = torch.cat((hf, hb), dim=-1)  # (batch, seq, input_size)

        # TODO: torch.cat(u, head) where u is stack representation.
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
            self._attn = a
            self._gate = g
        return c


class LatentFactorComposition(nn.Module):
    """A latent factor model for composition function."""
    def __init__(self, num_factors, input_size, num_layers, dropout, binary=False, device=None,
                 start_temp=1.0, min_temp=0.5, rate=0.09):
        super(LatentFactorComposition, self).__init__()
        assert input_size % 2 == 0, 'hidden size must be even'
        self.device = device
        self.binary = binary
        self.generative = nn.Linear(num_factors, input_size, bias=False)
        self.inference = nn.Sequential(
            BiRecurrentComposition(input_size, num_layers, dropout, device=device),
            nn.ReLU(),
            nn.Linear(input_size, num_factors))

        self.encoder = BiRecurrentComposition(input_size, num_layers, dropout, device=device)
        self.linear = nn.Linear(input_size, num_factors)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.annealer = AnnealTemperature(
            start_temp, min_temp, rate)

    def encode(self, head, children):
        h = self.encoder(head, children)
        return self.linear(self.relu(h))
        # return self.inference(head, children)

    def decode(self, x):
        return self.generative(x)

    def sample(self, alpha):
        temp = self.annealer.temp()
        if self.binary:
            if self.training:
                return BinaryConcrete(alpha, temp).sample()
            else:
                return (alpha > 0.5).float()  # argmax for binary variable
        else:
            if self.training:
                return Concrete(alpha, temp).sample()
            else:
                return dist.OneHotCategorical(logits=alpha).sample()

    def forward(self, head, children):
        """Forward pass that works for unpadded, i.e. equal length, batches.
        Args:
            children (torch.tensor): shape (batch, seq, dim)
            head (torch.tensor): shape (batch, dim)
        """
        alpha = self.encode(head, children)
        sample = self.sample(alpha)
        self._alpha, self._sample = alpha, sample  # store internally
        return self.decode(sample)

    def kl(self, alpha):
        if self.binary:
            kl = (self.softmax(alpha) *
                (alpha - torch.log(torch.tensor(1.0) / 2.0))).sum()
        else:
            kl = (self.softmax(alpha) *
                (alpha - torch.log(torch.tensor(1.0) / alpha.size(-1)))).sum()
        return kl


class LatentAttentionComposition(nn.Module):
    """A latent attention (`hard attention`) composition function."""
    def __init__(self, input_size, num_layers, dropout, device=None):
        pass
