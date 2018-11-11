import os
import json
from collections import Counter, defaultdict

from actions import SHIFT, REDUCE, NT, GEN, is_gen
from tree import fromstring
from utils import replace_quotes, replace_brackets


UNK = '<UNK>'


class Vocabulary:

    def __init__(self, unk=False):
        self._values = []
        self._indices = {}
        self._counts = defaultdict(int)
        self.unk = unk

    def __iter__(self):
        return iter(self._values)

    def add(self, value):
        self._counts[value] += 1
        if not value in self._values:
            self._values.append(value)
            self._indices[value] = len(self._values) - 1

    def value(self, index):
        assert 0 <= index < len(self._values)
        return self._values[index]

    def index(self, value):
        if value in self._indices:
            return self._indices[value]
        else:
            assert self.unk
            return self._indices[UNK]

    def count(self, value):
        if value in self._counts:
            return self._counts[value]
        else:
            assert self.unk
            return self._counts[UNK]

    def values(self, indices):
        return [self.value(index) for index in indices]

    def indices(self, values):
        return [self.index(value) for value in values]

    def process(self, words):
        """Replace unknown words with UNK."""
        return self.values(self.indices(words))

    def save(self, path):
        path = path + '.json' if not path.endswith('.json') else path
        with open(path, 'w') as f:
            json.dump(self._indices, f, indent=4)

    def load(self, path):
        with open(path, 'w') as f:
            self._indices = json.load(f)

    @staticmethod
    def fromlist(values, unk=False):
        self = Vocabulary()
        self._values = list(sorted(set(values)))
        self._indices = dict((value, i) for i, value in enumerate(self))
        self._counts = defaultdict(int, Counter(values))
        self.unk = unk
        return self

    @property
    def size(self):
        return len(self._values)

    @property
    def unks(self):
        return [value for value in self._values if value.startswith(UNK)]


class LanguageModelDataset:

    def __init__(self, vocab, path):
        self.words = []
        self.read(vocab, path)

    def read(self, vocab, path):
        with open(path) as f:
            for i, line in enumerate(f):
                words = replace_brackets(replace_quotes(line.strip().split()))
                self.words.append(vocab.indices(words))

    @property
    def data(self):
        return self.words


if __name__ == '__main__':

    SemiSupervisedCorpus(
        '../data/train/ptb.train.oracle',
        '../data/dev/ptb.dev.oracle',
        '../data/test/ptb.test.oracle',
        '/Users/daan/data/one-billion-words/heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100'
    )
