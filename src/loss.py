import torch
import torch.nn as nn
import numpy as np

from data import wrap


class LossCompute:
    def __init__(self, criterion, device):
        self.criterion = criterion()
        self.device = device
        self._train_losses = []
        self._reset()

    def _reset(self):
        self._loss = []

    def __call__(self, logits, y):
        assert isinstance(logits, torch.Tensor), logits
        assert isinstance(y, int), y
        assert logits.size(0) == 1, logits.size()
        assert logits.size(1) == len(y), logits.size()
        y = wrap([y], self.device)
        loss = self.criterion(logits, y)
        self._loss.append(loss)
        return loss

    def get_loss(self, update=None):
        """Compute the total loss."""
        loss = sum(self._loss)
        self._reset()
        return loss


class ElboCompute(LossCompute):
    def __init__(self, criterion, device,
                 anneal=True, anneal_method='logistic', anneal_step=2.5e-3, anneal_rate=500):
        super(ElboCompute, self).__init__(criterion, device)
        self.anneal = anneal
        self._train_kl = []
        self._reset()
        if anneal:
            self.annealer = AnnealKL(
                method=anneal_method, step=anneal_step, rate=anneal_rate)

    def _reset(self):
        super(ElboCompute, self)._reset()
        self._kl = []

    def add_kl(self, kl):
        self._kl.append(kl)

    def get_loss(self, update):
        """Compute the loss (negative elbo)."""
        assert len(self._kl) > 0, 'not added any kl terms'
        if self.anneal:
            alpha = self.annealer.alpha()
        else:
            alpha = 1.0
        loss, kl = sum(self._loss), sum(self._kl)
        neg_elbo = loss + alpha*kl
        self._train_losses.append(loss.item())
        self._train_kl.append(kl.item())
        self._reset()
        return neg_elbo


class AnnealKL:
    """Anneal the KL in an ELBO objective."""
    def __init__(self, method='logistic', step=2.5e-3, rate=2500):
        assert method in ('logistic', 'linear')
        self.method = method
        self.rate = rate
        self.step = step
        self.i = 0

    def alpha(self):
        self.i += 1
        if self.method == 'logistic':
            alpha = float(1 / (1 + np.exp(-self.step*(self.i - self.rate))))
        if self.method == 'linear':
            alpha = min(1, self.i / self.rate)
        self._alpha = alpha
        return alpha


class AnnealTemperature:
    def __init__(self, temp_interval=300, start_temp=1.0, min_temp=0.5, rate=0.00009):
        self.temp_interval = temp_interval
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.rate = rate
        self.i = 0

    def temp(self):
        self.i += 1
        if self.i % self.temp_interval == 0:
            temp = np.exp(-self.start_temp*self.i*self.rate)
        self._temp = temp
        return max(temp, self.min_temp)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    annealer = AnnealKL(method='logistic')

    alphas = [annealer(step) for step in range(5000)]
    plt.plot(alphas)
    plt.savefig('anneal-alphas.pdf')
