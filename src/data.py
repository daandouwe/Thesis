import sys
import os
from collections import defaultdict

import torch
from torch.autograd import Variable
import numpy as np

from get_configs import EMPTY_TOKEN, SEPARATOR

PAD_TOKEN = '-PAD-'
PAD_INDEX = 0

EMPTY_INDEX = 1

def pad(batch):
    """
    Pad a batch of irregular length indices and wrap it.
    """
    lens = list(map(len, batch))
    max_len = max(lens)
    padded_batch = []
    for k, seq in zip(lens, batch):
        padded =  seq + (max_len - k)*[PAD_INDEX]
        padded_batch.append(padded)
    return wrap(padded_batch)

def wrap(batch):
    """
    Packages the batch as a Variable containing a LongTensor
    so the batch is ready as input for a PyTorch model.
    """
    return Variable(torch.LongTensor(batch))


class Dictionary:
    """
    A dictionary for stack, buffer, and action symbols.
    """
    def __init__(self, path):
        self.s2i = dict()
        self.w2i = dict()
        self.a2i = dict()

        self.i2s = []
        self.i2w = []
        self.i2a = []

        self.initialize()
        self.read(path)

    def initialize(self):
        self.s2i[PAD_TOKEN] = 0
        self.w2i[PAD_TOKEN] = 0
        self.a2i[PAD_TOKEN] = 0

        self.s2i[EMPTY_TOKEN] = 1
        self.w2i[EMPTY_TOKEN] = 1
        self.a2i[EMPTY_TOKEN] = 1

        self.i2s.append(PAD_TOKEN)
        self.i2w.append(PAD_TOKEN)
        self.i2a.append(PAD_TOKEN)

        self.i2s.append(EMPTY_TOKEN)
        self.i2w.append(EMPTY_TOKEN)
        self.i2a.append(EMPTY_TOKEN)

    def read(self, path):
        with open(path + '.stack', 'r') as f:
            for i, line in enumerate(f):
                s = line.rstrip()
                self.s2i[s] = i
                self.i2s.append(s)
        with open(path + '.vocab', 'r') as f:
            for i, line in enumerate(f):
                w = line.rstrip()
                self.w2i[w] = i
                self.i2w.append(w)
        with open(path + '.actions', 'r') as f:
            for i, line in enumerate(f):
                a = line.rstrip()
                self.a2i[a] = i
                self.i2a.append(a)

class Data:
    """
    A dataset with parse configurations.
    """
    def __init__(self, path, dictionary):
        self.stack = []
        self.buffer = []
        self.history = []
        self.action = []
        self.lengths = []

        self.read(path, dictionary)

    def read(self, path, dictionary):
        with open(path, 'r') as f:
            for line in f:
                stack, buffer, history, action = line.split(SEPARATOR)
                stack = [dictionary.s2i[s] for s in stack.split()]
                buffer = [dictionary.w2i[w] for w in buffer.split()]
                history = [dictionary.a2i[a] for a in history.split()]
                action = dictionary.a2i[action.strip()]
                self.stack.append(stack)
                self.buffer.append(buffer)
                self.history.append(history)
                self.action.append(action)
        self.lengths = [len(l) for l in self.stack]

    def _reorder(self, new_order):
        self.stack = [self.stack[i] for i in new_order]
        self.buffer = [self.buffer[i] for i in new_order]
        self.history = [self.history[i] for i in new_order]
        self.action = [self.action[i] for i in new_order]
        self.lengths = [self.lengths[i] for i in new_order]

    def order(self):
        old_order = zip(range(len(self.lengths)), self.lengths)
        new_order, _ = zip(*sorted(old_order, key=lambda t: t[1]))
        self._reorder(new_order)

    def shuffle(self):
        n = len(self.stack)
        new_order = list(range(0, n))
        np.random.shuffle(new_order)
        self._reorder(new_order)

    def batches(self, batch_size, shuffle=True, length_ordered=False):
        """
        An iterator over batches.
        """
        n = len(self.stack)
        batch_order = list(range(0, n, batch_size))
        if shuffle:
            self.shuffle()
            np.random.shuffle(batch_order)
        if length_ordered:
            self.order()
        for i in batch_order:
            stack = pad(self.stack[i:i+batch_size])
            buffer = pad(self.buffer[i:i+batch_size])
            history = pad(self.history[i:i+batch_size])
            action = wrap(self.action[i:i+batch_size])
            yield stack, buffer, history, action

class Corpus:
    """
    A corpus of three datasets (train, development, and test) and a dictionary.
    """
    def __init__(self, data_path="../tmp/ptb"):
        self.dictionary = Dictionary(data_path)
        self.train = Data(data_path + '.configs', self.dictionary)
        # self.dev = Data(os.path.join(data_path, "dev-stanford-raw.conll"), self.dictionary)
        # self.test = Data(os.path.join(data_path, "test-stanford-raw.conll"), self.dictionary)

if __name__ == "__main__":
    # Example usage:
    corpus = Corpus(data_path="../tmp/ptb")
    batches = corpus.train.batches(4, length_ordered=True)
    for _ in range(3):
        stack, buffer, history, action = next(batches)
        print(stack, buffer, history, action)
