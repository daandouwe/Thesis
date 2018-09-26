import sys
import os
import string
from tqdm import tqdm

import torch
import numpy as np

from datatypes import Token, Word, Nonterminal, Action
from actions import SHIFT, REDUCE, NT, GEN


PAD_CHAR = '_'
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
    tensor = torch.tensor(batch, device=device, dtype=torch.long)
    return tensor.to(device)


def get_sentences(path):
    """Chunks the oracle file into sentences."""
    def get_sent_dict(sent):
        d = {
                'tree'    : sent[0],
                'tags'    : sent[1],
                'upper'   : sent[2],
                'lower'   : sent[3],
                'unked'   : sent[4],
                'actions' : sent[5:]
            }
        return d

    sentences = []
    with open(path) as f:
        sent = []
        for line in f:
            if line == '\n':
                sentences.append(sent)
                sent = []
            else:
                sent.append(line.rstrip())
        return [get_sent_dict(sent) for sent in sentences if sent]


class Dictionary:
    """A dictionary for stack, buffer, and action symbols."""
    def __init__(self, path, name, use_chars=False):
        self.n2i = dict()  # nonterminals
        self.w2i = dict()  # words
        self.i2n = []
        self.i2w = []
        self.use_chars = use_chars
        self.initialize()
        self.read(path, name)

    def initialize(self):
        self.w2i[PAD_CHAR] = PAD_INDEX
        self.i2w.append(PAD_CHAR)

    def read(self, path, name):
        with open(os.path.join(path, name + '.vocab'), 'r') as f:
            start = len(self.w2i)
            if self.use_chars:
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
    def __init__(self, path, dictionary, model, textline, use_chars=False, max_lines=-1):
        self.dictionary = dictionary
        self.sentences = []
        self.actions = []
        self.use_chars = use_chars
        self.model = model
        self.read(path, dictionary, textline, max_lines)

    def __str__(self):
        return f'{len(self.sentences):,} sentences'

    def _order(self, new_order):
        self.sentences = [self.sentences[i] for i in new_order]
        self.actions = [self.actions[i] for i in new_order]

    def read(self, path, dictionary, textline, max_lines):
        sents = get_sentences(path)  # a list of dict
        nlines = len(sents)
        for i, sent_dict in enumerate(tqdm(sents)):
            if max_lines > 0 and i > max_lines:
                break
            # Get sentence items.
            original, processed = sent_dict['upper'].split(), sent_dict[textline].split()
            sentence = [Token(orig, proc) for orig, proc in zip(original, processed)]
            sentence_items = []
            for token in sentence:
                if self.use_chars:
                    index = [dictionary.w2i[char] for char in token.processed]
                else:
                    index = dictionary.w2i[token.processed]
                sentence_items.append(Word(token, index))
            # Get action items.
            actions = sent_dict['actions']
            action_items = []
            token_idx = 0
            for a in actions:
                if a == 'SHIFT':
                    if self.model == 'disc':
                        action = Action('SHIFT', Action.SHIFT_INDEX)
                    if self.model == 'gen':
                        token = sentence[token_idx]
                        action = GEN(Word(token, dictionary.w2i[token.processed]))
                        token_idx += 1
                elif a == 'REDUCE':
                    action = Action('REDUCE', Action.REDUCE_INDEX)
                elif a.startswith('NT'):
                    nt = a[3:-1]
                    action = NT(Nonterminal(nt, dictionary.n2i[nt]))
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
    def __init__(self, data_path='../data', model='disc', textline='unked', name='ptb', use_chars=False, max_lines=-1):
        self.dictionary = Dictionary(path=os.path.join(data_path, 'vocab', textline), name=name, use_chars=use_chars)
        self.train = Data(path=os.path.join(data_path, 'train', name + '.train.oracle'),
                        dictionary=self.dictionary, model=model, textline=textline, use_chars=use_chars, max_lines=max_lines)
        self.dev = Data(path=os.path.join(data_path, 'dev', name + '.dev.oracle'),
                        dictionary=self.dictionary, model=model, textline=textline, use_chars=use_chars)
        self.test = Data(path=os.path.join(data_path, 'test', name + '.test.oracle'),
                        dictionary=self.dictionary, model=model, textline=textline, use_chars=use_chars)

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
    corpus = Corpus(data_path='../data', textline='unked', use_chars=False)
    batches = corpus.test.batches(1, length_ordered=False)
    sentence, actions = batches[0]
    print([word.token for word in sentence])
    print([action.token for action in actions])

    with open('w2i.json', 'w') as f:
        json.dump(corpus.dictionary.w2i, f, sort_keys=True, indent=4)
    with open('n2i.json', 'w') as f:
        json.dump(corpus.dictionary.n2i, f, sort_keys=True, indent=4)
