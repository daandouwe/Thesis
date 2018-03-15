import torch
import torch.nn as nn
from torch.autograd import Variable

class EncoderBoW(nn.Module):
    """
    Bag of Words (BoW) inference network ('encoder') for EmbedAlign model.
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim, z_dim):
        super(EncoderBoW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.linear = nn.Linear(emb_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.sigma = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.embedding(x)
        h = self.softplus(self.linear(x))
        mu = self.mu(h)
        sigma = self.softplus(self.sigma(h))
        return mu, sigma


class Decoder(nn.Module):
    """
    Generative network ('decoder') for EmbedAlign.
    One affine layer followed by (log)softmax.
    """
    def __init__(self, z_dim, vocab_size):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(z_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, z, log=False):
        if log:
            return self.logsoftmax(self.linear(z))
        else:
            return self.softmax(self.linear(z))


class EmbedAlign(nn.Module):
    """
    The EmbedAlign model as described in https://arxiv.org/abs/1802.05883.
    """
    def __init__(self, l1_vocab_size, l2_vocab_size, emb_dim, hidden_dim, z_dim):
        super(EmbedAlign, self).__init__()
        self.encoder = EncoderBoW(l1_vocab_size, emb_dim, hidden_dim, z_dim)
        self.f = Decoder(z_dim, l1_vocab_size)
        self.g = Decoder(z_dim, l2_vocab_size)

    def sample(self, mu, sigma):
        """
        Returns a sample from N(mu, sigma)
        """
        normal = torch.distributions.Normal(torch.zeros(mu.shape), torch.ones(sigma.shape))
        eps = Variable(normal.sample())
        z = mu + eps * sigma
        return z

    def kl(self, mu, sigma):
        """
        Computes the KL-divergence between N(mu, sigma*I) and N(0, I)
        """
        return torch.mean(torch.mean(0.5 * torch.sum(sigma + mu**2 - 1. - torch.log(sigma), dim=-1), dim=-1))

    def log_px(self, x, px):
        """
        Computes log P(x|z).
        """
        px = torch.gather(px, -1, x.unsqueeze(-1)).squeeze()
        mask = (x > 0).float()
        return torch.mean(torch.mean(mask * px, dim=-1))
        # return torch.mean(torch.sum(x_mask * px, dim=-1))

    def log_py(self, y, py, m):
        """
        Computes log P(y|z,a) by marginalizing alignments a.
        """
        indices = y.unsqueeze(-1).expand(-1, -1, m).transpose(1, 2) # [batch_size, m, n]
        selected = torch.gather(py, -1, indices)
        marginal = torch.log(torch.mean(selected, dim=1)) # Marginalize alignments
        mask = (y > 0).float()
        return torch.mean(torch.mean(mask * marginal, dim=-1))
        # return torch.mean(torch.sum(y_mask * y_marginal, dim=-1))

    def forward(self, x, y):
        """
        A full forward pass to compute the Elbo.
        Compute posterior q(z|x) (with 'encoder'), sample one z, and use it to
        approximate expected log likelihoods p(x|z) and p(y|z) under q(z|x).
        Compute KL of q(z|x) with prior.
        """
        mu, sigma = self.encoder(x)
        z = self.sample(mu, sigma)
        print(x.shape)
        print(y.shape)
        print(z.shape)

        px = self.f(z, log=True) # [batch_size, sent_len, l1_vocab_size]
        py = self.g(z)           # [batch_size, sent_len, l2_vocab_size]

        print(px.shape)
        print(py.shape)

        log_px = self.log_px(x, px)
        log_py = self.log_py(y, py, m=x.size(1))
        kl = self.kl(mu, sigma)

        return log_px, log_py, kl
