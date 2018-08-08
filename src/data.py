import sys
import os
import string
from collections import defaultdict
from tqdm import tqdm

import torch
from torch.autograd import Variable
import numpy as np

from scripts.get_vocab import get_sentences

PAD_TOKEN = '_PAD_'
EMPTY_TOKEN = '_EMPTY_' # used as dummy to encode an empty stack or history
REDUCED_TOKEN = '_REDUCED_' # used as dummy for reduced sequences

PAD_INDEX = 0
EMPTY_INDEX = 1
REDUCED_INDEX = 2

def pad(batch):
    """Pad a batch of irregular length indices."""
    lens = list(map(len, batch))
    max_len = max(lens)
    padded_batch = []
    for k, seq in zip(lens, batch):
        padded =  seq + (max_len - k)*[PAD_INDEX]
        padded_batch.append(padded)
    return padded_batch

def wrap(batch, device):
    """Packages the batch as a Variable containing a LongTensor."""
    assert isinstance(batch, list)
    if isinstance(batch[0], list) and len(batch) > 1:
        batch = pad(batch)
    x = torch.LongTensor(batch)
    return x.to(device)

class Item:
    def __init__(self, token, index, embedding=None, encoding=None, nonterminal=False):
        """Make a data item.

        Args:
            token (str): a word, action, or nonterminal.
            index (int or [int]): an integer index for word embedding,
                or a list of integers if character embeddings are used.
            embedding (torch tensor): an embedding of the token.
            encoding (torch tensor): an encoding of the token.
        """
        self.token = token
        self.index = index
        self.embedding = embedding
        self.encoding = encoding
        self._nonterminal = nonterminal

    def __str__(self):
        return self.token

    @property
    def is_nonterminal(self):
        return self._nonterminal

class Action(Item):
    def __init__(self, token, index, embedding=None, encoding=None, symbol=None):
        super(Action, self).__init__(token, index, embedding, encoding)
        if symbol is not None:
            assert isinstance(symbol, Item)
        self.symbol = symbol

    @property
    def is_nonterminal(self):
        return self.symbol is not None

class Dictionary:
    """A dictionary for stack, buffer, and action symbols."""
    def __init__(self, path, char=False):
        self.n2i = dict() # nonterminals
        self.w2i = dict() # words
        self.a2i = dict() # actions

        self.i2n = []
        self.i2w = []
        self.i2a = []

        self.char = char

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
        with open(os.path.join(path, 'ptb.vocab'), 'r') as f:
            start = len(self.w2i)
            if self.char:
                chars = set(f.read())
                printable = set(string.printable)
                chars = list(chars | printable)
                for i, w in enumerate(chars):
                    self.w2i[w] = i
                    self.i2w.append(w)
            else:
                for i, line in enumerate(f, start):
                    w = line.rstrip()
                    self.w2i[w] = i
                    self.i2w.append(w)
        with open(os.path.join(path, 'ptb.nonterminals'), 'r') as f:
            for i, line in enumerate(f):
                s = line.rstrip()
                self.n2i[s] = i
                self.i2n.append(s)
        with open(os.path.join(path, 'ptb.actions'), 'r') as f:
            for i, line in enumerate(f):
                a = line.rstrip()
                self.a2i[a] = i
                self.i2a.append(a)

    @property
    def unks(self, unk_start='UNK'):
        return [w for w in self.w2i if w.startswith(unk_start)]

    @property
    def num_words(self):
        return len(self.w2i)

    @property
    def num_actions(self):
        return len(self.a2i)

    @property
    def num_nonterminals(self):
        return len(self.n2i)


class Data:
    """A dataset with parse configurations."""
    def __init__(self, path, dictionary, textline, char=False):
        self.dictionary = dictionary

        self.sentences = [] # each sentence as list of words
        self.actions = [] # each sequence of actions as list of indices

        self.char = char
        self.read(path, dictionary, textline)

    def __str__(self):
        return f'{len(self.sentences):,} sentences'

    def _order(self, new_order):
        self.sentences = [self.sentences[i] for i in new_order]
        self.actions = [self.actions[i] for i in new_order]

    def read(self, path, dictionary, textline):
        sents = get_sentences(path) # a list of `sent_dict` objects
        nlines = len(sents)
        for i, sent_dict in enumerate(tqdm(sents)):
            # Get sentence items.
            sentence = sent_dict[textline].split()
            sentence_items = []
            for token in sentence:
                if self.char:
                    index = [dictionary.w2i[char] for char in token]
                else:
                    index = dictionary.w2i[token]
                sentence_items.append(Item(token, index))
            # Get action items
            actions = sent_dict['actions']
            action_items = []
            for token in actions:
                if not token == 'SHIFT' and not token == 'REDUCE':
                    nonterminal = token[3:-1]
                    symbol = Item(nonterminal, self.dictionary.n2i[nonterminal])
                    token, index = 'OPEN', dictionary.a2i['OPEN']
                else:
                    index = dictionary.a2i[token]
                    symbol = None
                action_items.append(Action(token, index, symbol=symbol))
            # Store internally
            self.sentences.append(sentence_items)
            self.actions.append(action_items)
        self.lengths = [len(sent) for sent in self.sentences]

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
        batches = []
        for i in range(n):
            sentence = self.sentences[i]
            actions = self.actions[i]
            batches.append((sentence, actions))
        return batches

    @property
    def textline(self):
        return self.textline

class Corpus:
    """A corpus of three datasets (train, development, and test) and a dictionary."""
    def __init__(self, data_path='../tmp', textline='unked', char=False):
        self.dictionary = Dictionary(os.path.join(data_path, 'vocab', textline), char=char)
        self.train = Data(os.path.join(data_path, 'train', 'ptb.train.oracle'),
                        self.dictionary, textline, char=char)
        self.dev = Data(os.path.join(data_path, 'dev', 'ptb.dev.oracle'),
                        self.dictionary, textline, char=char)
        self.test = Data(os.path.join(data_path, 'test', 'ptb.test.oracle'),
                        self.dictionary, textline, char=char)

    def __str__(self):
        items = (
            'Corpus',
             f'vocab size: {self.dictionary.num_words:,}',
             f'train: {str(self.train)}',
             f'dev: {str(self.dev)}',
             f'test: {str(self.test)}',
        )
        return '\n'.join(items)

if __name__ == "__main__":
    # Example usage:
    corpus = Corpus(data_path='../tmp', textline='unked', char=False)
    batches = corpus.test.batches(1, length_ordered=False)
    sentence, actions = batches[0]
    print([item.token for item in sentence])
    print([item.token for item in actions])
    print([item.symbol.token if item.is_nonterminal else item.token for item in actions])
