#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.nn import functional as F


class BinaryConcrete:
    def __init__(self, alpha, temp):
        self.alpha = alpha
        self.gumbel = torch.distributions.Gumbel(
            torch.zeros(self.alpha.shape), torch.ones(self.alpha.shape))
        self.temp = temp
        self.sigmoid = nn.Sigmoid()

    def sample(self):
        return self.sigmoid((self.alpha + self.gumbel.sample()) / self.temp)


class Concrete:
    def __init__(self, alpha, temp):
        self.alpha = alpha
        self.gumbel = torch.distributions.Gumbel(
            torch.zeros(self.alpha.shape), torch.ones(self.alpha.shape))
        self.temp = temp
        self.softmax = nn.Softmax(dim=-1)

    def sample(self):
        return self.softmax((self.alpha + self.gumbel.sample()) / self.temp)
