import sys
import os
from collections import defaultdict

import torch
from torch.autograd import Variable
import numpy as np

from new_get_configs import get_sentences

PAD_TOKEN = '-PAD-'
EMPTY_TOKEN = '-EMPTY-'
PAD_INDEX = 0
EMPTY_INDEX = 1

def pad(batch, cuda=False, reverse=False):
    """Pad a batch of irregular length indices and wrap it."""
    lens = list(map(len, batch))
    max_len = max(lens)
    padded_batch = []
    if reverse:
        batch = [l[::-1] for l in batch]
    for k, seq in zip(lens, batch):
        padded =  seq + (max_len - k)*[PAD_INDEX]
        padded_batch.append(padded)
    return wrap(padded_batch, cuda=cuda)

def wrap(batch, cuda=False):
    """Packages the batch as a Variable containing a LongTensor."""
    if cuda:
        return Variable(torch.cuda.LongTensor(batch))
    else:
        return Variable(torch.LongTensor(batch))


class Dictionary:
    """A dictionary for stack, buffer, and action symbols."""
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

    def read(self, path, start=2):
        with open(path + '.stack', 'r') as f:
            for i, line in enumerate(f, start):
                s = line.rstrip()
                self.s2i[s] = i
                self.i2s.append(s)
        with open(path + '.vocab', 'r') as f:
            for i, line in enumerate(f, start):
                w = line.rstrip()
                self.w2i[w] = i
                self.i2w.append(w)
        with open(path + '.actions', 'r') as f:
            for i, line in enumerate(f, start):
                a = line.rstrip()
                self.a2i[a] = i
                self.i2a.append(a)

class Data:
    """A dataset with parse configurations."""
    def __init__(self, path, dictionary):
        self.sentences = []
        self.actions = []

        self.read(path, dictionary)

    def read(self, path, dictionary, print_every=1000):
        sents = get_sentences(path)
        nlines = len(sents)
        for i, sent_dict in enumerate(sents):
            sent = sent_dict['unked'].split()
            actions = sent_dict['actions']

            sent = [dictionary.w2i[w] for w in sent]
            actions = [dictionary.a2i[w] for w in actions]
            self.sentences.append(sent)
            self.actions.append(actions)
        self.lengths = [len(sent) for sent in self.sentences]

    def _reorder(self, new_order):
        self.sentences = [self.sentences[i] for i in new_order]
        self.actions = [self.actions[i] for i in new_order]

    def order(self):
        old_order = zip(range(len(self.lengths)), self.lengths)
        new_order, _ = zip(*sorted(old_order, key=lambda t: t[1]))
        self._reorder(new_order)

    def shuffle(self):
        n = len(self.sentences)
        new_order = list(range(0, n))
        np.random.shuffle(new_order)
        self._reorder(new_order)

    def batches(self, shuffle=True,
                length_ordered=False, cuda=False):
        """An iterator over batches."""
        n = len(self.sentences)
        if shuffle:
            self.shuffle()
        if length_ordered:
            self.order()
        for i in range(n):
            sentence = self.sentences[i]
            actions = self.actions[i]
            yield sentence, actions


class Corpus:
    """A corpus of three datasets (train, development, and test) and a dictionary."""
    def __init__(self, data_path="../tmp/ptb"):
        self.dictionary = Dictionary(data_path)
        self.train = Data(data_path + '.oracle', self.dictionary)

        # self.dev = Data(os.path.join(data_path, "dev-stanford-raw.conll"), self.dictionary)
        # self.test = Data(os.path.join(data_path, "test-stanford-raw.conll"), self.dictionary)

if __name__ == "__main__":
    # Example usage:
    corpus = Corpus(data_path="../tmp/ptb")
    batches = corpus.train.batches(1, length_ordered=True)
    for _ in range(2):
        sentence, actions = next(batches)
        print(sentence, actions)
