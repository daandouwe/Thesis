import sys
import os
from collections import defaultdict

import torch
from torch.autograd import Variable
import numpy as np
from gensim.models import KeyedVectors

from get_configs import get_sentences

PAD_TOKEN = '-PAD-'
EMPTY_TOKEN = '-EMPTY-'
REDUCED_TOKEN = '-REDUCED-' # used as dummy for reduced sequences
PAD_INDEX = 0
EMPTY_INDEX = 1
REDUCED_INDEX = 2

def wrap(batch, cuda=False):
    """Packages the batch as a Variable containing a LongTensor."""
    if cuda:
        return Variable(torch.cuda.LongTensor(batch))
    else:
        return Variable(torch.LongTensor(batch))

def load_glove(dictionary, dim=100, dir='~/glove'):
    assert dim in (50, 100, 200, 300), 'invalid dim: choose from (50, 100, 200, 300).'
    path = os.path.join(dir, 'glove.6B.{}d.gensim.txt'.format(dim))
    glove = KeyedVectors.load_word2vec_format(path, binary=False)
    vectors = []
    for w in dictionary.i2w:
        try:
            v = glove[w]
            vectors.append(v)
        except KeyError:
            print('word `{}` not found.'.format(w))
            vectors.append(np.zeros(dim)) # NOTE: Find better fix
    vectors = np.vstack(vectors)
    vectors = torch.FloatTensor(vectors)
    return vectors

class Dictionary:
    """A dictionary for stack, buffer, and action symbols."""
    def __init__(self, path):
        self.n2i = dict() # nonterminals
        self.w2i = dict() # words
        self.a2i = dict() # actions

        self.i2n = []
        self.i2w = []
        self.i2a = []

        self.initialize()
        self.read(path)

    def initialize(self):
        self.w2i[PAD_TOKEN] = 0
        self.w2i[EMPTY_TOKEN] = 1
        self.w2i[REDUCED_TOKEN] = 2

        self.i2w.append(PAD_TOKEN)
        self.i2w.append(EMPTY_TOKEN)
        self.i2w.append(REDUCED_TOKEN)

    def read(self, path):
        with open(path + '.vocab', 'r') as f:
            start = len(self.w2i)
            for i, line in enumerate(f, start):
                w = line.rstrip()
                self.w2i[w] = i
                self.i2w.append(w)
        with open(path + '.nonterminals', 'r') as f:
            for i, line in enumerate(f):
                s = line.rstrip()
                self.n2i[s] = i
                self.i2n.append(s)
        with open(path + '.actions', 'r') as f:
            for i, line in enumerate(f):
                a = line.rstrip()
                self.a2i[a] = i
                self.i2a.append(a)

class Data:
    """A dataset with parse configurations."""
    def __init__(self, path, dictionary):
        self.sentences = []
        self.indices = []
        self.actions = []

        self.read(path, dictionary)

    def read(self, path, dictionary, print_every=1000):
        sents = get_sentences(path)
        nlines = len(sents)
        for i, sent_dict in enumerate(sents):
            sent = sent_dict['unked'].split()
            actions = sent_dict['actions']

            ids = [dictionary.w2i[w] for w in sent]
            actions = [dictionary.a2i[w] for w in actions]
            self.sentences.append(sent)
            self.indices.append(ids)
            self.actions.append(actions)

        self.lengths = [len(sent) for sent in self.sentences]

    def _order(self, new_order):
        self.sentences = [self.sentences[i] for i in new_order]
        self.indices = [self.indices[i] for i in new_order]
        self.actions = [self.actions[i] for i in new_order]

    def order(self):
        old_order = zip(range(len(self.lengths)), self.lengths)
        new_order, _ = zip(*sorted(old_order, key=lambda t: t[1]))
        self._order(new_order)

    def shuffle(self):
        n = len(self.sentences)
        new_order = list(range(0, n))
        np.random.shuffle(new_order)
        self._order(new_order)

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
            ids = self.indices[i]
            actions = self.actions[i]
            yield sentence, ids, actions


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
