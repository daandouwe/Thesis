import sys
import os
import string
from tqdm import tqdm

import torch
import numpy as np

from datatypes import Word, Nonterminal
from actions import SHIFT, REDUCE, NT, GEN
from scripts.get_vocab import get_sentences


PAD_TOKEN = '_PAD_'
PAD_INDEX = 0


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
    if len(batch) > 1 and isinstance(batch[0], list):
        batch = pad(batch)
    return torch.tensor(batch, device=device, dtype=torch.long)


class Dictionary:
    """A dictionary for stack, buffer, and action symbols."""
    def __init__(self, path, name, char=False):
        self.n2i = dict()  # nonterminals
        self.w2i = dict()  # words
        self.i2n = []
        self.i2w = []
        self.char = char
        self.initialize()
        self.read(path, name)

    def initialize(self):
        self.w2i[PAD_TOKEN] = PAD_INDEX
        self.i2w.append(PAD_TOKEN)

    def read(self, path, name):
        with open(os.path.join(path, name + '.vocab'), 'r') as f:
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
        with open(os.path.join(path, name + '.nonterminals'), 'r') as f:
            start = len(self.n2i)
            for i, line in enumerate(f, start):
                s = line.rstrip()
                self.n2i[s] = i
                self.i2n.append(s)

    @property
    def unks(self, unk_start='UNK'):
        return [w for w in self.w2i if w.startswith(unk_start)]

    @property
    def num_words(self):
        return len(self.w2i)

    @property
    def num_nonterminals(self):
        return len(self.n2i)


class Data:
    """A dataset with parse configurations."""
    def __init__(self, path, dictionary, textline, char=False):
        self.dictionary = dictionary
        self.sentences = []
        self.actions = []
        self.char = char
        self.read(path, dictionary, textline)

    def __str__(self):
        return f'{len(self.sentences):,} sentences'

    def _order(self, new_order):
        self.sentences = [self.sentences[i] for i in new_order]
        self.actions = [self.actions[i] for i in new_order]

    def read(self, path, dictionary, textline):
        sents = get_sentences(path)  # a list of dict
        nlines = len(sents)
        for i, sent_dict in enumerate(tqdm(sents)):  # tqdm: a progressbar
            # Get sentence items.
            sentence = sent_dict[textline].split()
            sentence_items = []
            for token in sentence:
                if self.char:
                    index = [dictionary.w2i[char] for char in token]
                else:
                    index = dictionary.w2i[token]
                sentence_items.append(Word(token, index))
            # Get action items.
            actions = sent_dict['actions']
            action_items = []
            for token in actions:
                if token == SHIFT.token:
                    action = SHIFT
                elif token == REDUCE.token:
                    action = REDUCE
                elif token.startswith('NT'):
                    nt = token[3:-1]
                    nt = Nonterminal(nt, dictionary.n2i[nt])
                    action = NT(nt)
                elif token.startswith('GEN'):
                    word = token[4:-1]
                    word = Word(word, dictionary.w2i[word])
                    action = GEN(word)
                action_items.append(action)
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
    def __init__(self, data_path='../tmp', textline='unked', name='ptb', char=False):
        self.dictionary = Dictionary(path=os.path.join(data_path, 'vocab', textline), name=name, char=char)
        self.train = Data(path=os.path.join(data_path, 'train', name + '.train.oracle'),
                        dictionary=self.dictionary, textline=textline, char=char)
        self.dev = Data(path=os.path.join(data_path, 'dev', name + '.dev.oracle'),
                        dictionary=self.dictionary, textline=textline, char=char)
        self.test = Data(path=os.path.join(data_path, 'test', name + '.test.oracle'),
                        dictionary=self.dictionary, textline=textline, char=char)

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
    import json

    # Example usage:
    corpus = Corpus(data_path='../tmp', textline='unked', char=False)
    batches = corpus.test.batches(1, length_ordered=False)
    sentence, actions = batches[0]
    print([word.token for word in sentence])
    print([action.token for action in actions])

    with open('w2i.json', 'w') as f:
        json.dump(corpus.dictionary.w2i, f, sort_keys=True, indent=4)
    with open('n2i.json', 'w') as f:
        json.dump(corpus.dictionary.n2i, f, sort_keys=True, indent=4)
