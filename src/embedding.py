import torch
import torch.nn as nn
from torch.autograd import Variable

from data import EMPTY_INDEX, REDUCED_INDEX

class ConvolutionalCharEmbedding(nn.Module):
    """Convolutional character embedding following https://arxiv.org/pdf/1508.06615.pdf."""
    def __init__(self, nchars, emb_dim, padding_idx,
                 max_kernel_width=6, char_emb_dim=15, filter_factor=25, activation='Tanh',
                 dropout=0, device=None):
        super(ConvolutionalCharEmbedding, self).__init__()
        self.padding_idx = padding_idx
        self.device = device
        self.max_kernel_width = max_kernel_width
        # Embedding for characters.
        self.embedding = nn.Embedding(nchars, char_emb_dim, padding_idx=padding_idx)
        # For special tokens we learn embeddings directly.
        special = (EMPTY_INDEX, REDUCED_INDEX)
        self.er_embedding = nn.Embedding(max(special), emb_dim)

        filter_size = lambda kernel_size: filter_factor * kernel_size
        kernel_sizes = range(1, max_kernel_width+1)
        self.conv_size = sum(map(filter_size, kernel_sizes))
        self.convs = nn.ModuleList(
            [nn.Conv1d(char_emb_dim, filter_size(i), kernel_size=i)
                for i in kernel_sizes]
        )
        self.linear = nn.Linear(self.conv_size, emb_dim)

        self.act_fn = getattr(nn, activation)()
        self.pool = nn.AdaptiveMaxPool1d(1) # Max pooling over time.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """Expect input of shape (sent_len, word_len).

        Note: Works specifically for batches of size 1.
        """
        if x.size(0) == 1:
            # x = tensor([EMPTY_INDEX]) or x = tensor([REDUCED_INDEX])
            return self.er_embedding(x)

        if x.size(1) < self.max_kernel_width:
            # Add padding when word is shorter than max conv kernel.
            rest = self.max_kernel_width - x.size(1)
            pad = torch.zeros(x.size(0), rest, device=self.device).long()
            x = torch.cat((x, pad), dim=-1)

        # Preprocessing of character batch.
        sent_len, word_len = x.shape
        mask = (x != self.padding_idx).float()
        x = self.embedding(x)   # (sent, word, emb)
        mask = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
        x = mask * x
        x = x.transpose(1, 2)   # (sent, emb, word)

        f = [self.pool(self.act_fn(conv(x))).squeeze(-1) # (sent, filter_size)
                for conv in self.convs]

        f = torch.cat(f, dim=-1) # (sent, conv_size)
        f = self.dropout(f)
        f = self.act_fn(self.linear(f)) # (sent, emb_dim)

        return f.contiguous().view(sent_len, f.size(-1)) # (sent, emb_dim)
