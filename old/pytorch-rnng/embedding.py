import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvolutionalCharEmbedding(nn.Module):
    """Convolutional character embedding.

     Follows `Character-Aware Neural Language Models`,
     source: https://arxiv.org/pdf/1508.06615.pdf.
     """
    def __init__(self, nchars, emb_dim, padding_idx,
                 max_kernel_width=6, char_emb_dim=15,
                 filter_factor=25, activation='Tanh',
                 dropout=0, device=None):
        super(ConvolutionalCharEmbedding, self).__init__()
        self.padding_idx = padding_idx
        self.device = device
        self.max_kernel_width = max_kernel_width
        self.embedding_dim = emb_dim
        self.embedding = nn.Embedding(nchars, char_emb_dim, padding_idx=padding_idx)

        filter_size = lambda kernel_size: filter_factor * kernel_size
        kernel_sizes = range(1, max_kernel_width+1)
        self.conv_size = sum(map(filter_size, kernel_sizes))
        self.convs = nn.ModuleList(
            [nn.Conv1d(char_emb_dim, filter_size(i), kernel_size=i) for i in kernel_sizes]
        )
        self.linear = nn.Linear(self.conv_size, emb_dim)

        self.act_fn = getattr(nn, activation)()
        self.pool = nn.AdaptiveMaxPool1d(1)  # Max pooling over time.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """Expect input of shape (sent_len, word_len).

        Note: Works only for batches of size 1.
        """
        if x.size(1) < self.max_kernel_width:
            # Add padding when word is shorter than max conv kernel.
            rest = self.max_kernel_width - x.size(1)
            pad = torch.zeros(x.size(0), rest, device=self.device).long()
            x = torch.cat((x, pad), dim=-1)

        # Preprocessing of character batch.
        sent_len, word_len = x.shape
        mask = (x != self.padding_idx).float()
        x = self.embedding(x)  # (sent, word, emb)
        mask = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
        x = mask * x
        x = x.transpose(1, 2)  # (sent, emb, word)

        f = [self.pool(self.act_fn(conv(x))).squeeze(-1)  # (sent, filter_size)
             for conv in self.convs]

        f = torch.cat(f, dim=-1)  # (sent, conv_size)
        f = self.dropout(f)
        f = self.act_fn(self.linear(f))  # (sent, emb_dim)

        return f.contiguous().view(sent_len, f.size(-1))  # (sent, emb_dim)


class FineTuneEmbedding(nn.Module):
    """Fine-tunes a pretrained Embedding layer.

    Follows the method described in Yoav Goldberg's `Neural
    Network Methods for Natural LanguageProcessing` (p. 117).
    In particular we let the fine-tuned embeddings W' be
    computed as

        W' = W + Δ

    were W are the pre-trained embeddings and Δ the fine-tuning
    weights. W is fixed, and the weights in Δ are initialized
    to zero and trained with an L2 penalty.

    Also: see this https://twitter.com/adveisner/status/896428540538834944
    and this https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting.

    Note: the average norm of the pretrained GloVe embeddings is ~0.03,
    and if, say, we'd like our fine-tuning to be at most a tenth of that,
    e.g. of the order 0.001, then 100 * 0.001 = 0.1 is the additional
    loss we incur. This seems sensible to me.
    """
    def __init__(self, embedding, weight_decay=100):
        super(FineTuneEmbedding, self).__init__()
        assert isinstance(embedding, nn.Embedding)
        num_embeddings, embedding_dim = embedding.weight.shape
        self.embedding = embedding
        self.delta = nn.Embedding(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight_decay = weight_decay

        # Freeze the pre-trained weights.
        self.embedding.weight.requires_grad = False
        # Initilize the fine-tuning additions with zeros.
        self.delta.weight.data.zero_()

    def forward(self, input):
        """Return W'[input] where W' = W + Δ."""
        return self.embedding(input) + self.delta(input)

    def delta_norm(self):
        """Return the (average) L2 norm of Δ."""
        # We average over vocabulary, otherwise we are
        # not consistent accross varying vocab-sizes.
        avg_norm = torch.norm(
            self.delta.weight, p=2) / self.num_embeddings
        return avg_norm.squeeze()

    def delta_penalty(self):
        """Return the (average) L2 norm of Δ scaled by the weight-decay term."""
        return self.weight_decay * self.delta_norm()
