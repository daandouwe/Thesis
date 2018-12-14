import os
import json
from collections import Counter, defaultdict

import numpy as np

from actions import SHIFT, REDUCE, NT, GEN, is_gen
from tree import fromstring
from utils import unkify


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
            return 0

    def index_or_unk(self, value, unk_value):
        if value in self._indices:
            return self._indices[value]
        else:
            assert unk_value in self._indices, f'unk not in vocab `{unk_value}`'
            return self._indices[unk_value]

    def values(self, indices):
        return [self.value(index) for index in indices]

    def indices(self, values):
        return [self.index(value) for value in values]

    def process(self, words):
        """Replaces unkown words with unk."""
        return self.values(self.indices(words))

    def unkify(self, words):
        """Dynamic unking used during training."""
        assert self.unk, 'vocab has no unk'
        unked = []
        for i, word in enumerate(words):
            count = self.count(word)
            if not count or np.random.rand() < 1 / (1 + count):
                unked.append(UNK)
            else:
                unked.append(word)
        return unked

    def save(self, path):
        # TODO: do this for everything
        path = path + '.json' if not path.endswith('.json') else path
        with open(path, 'w') as f:
            json.dump(self._indices, f, indent=4)

    def load(self, path):
        # TODO: do this for everything
        with open(path, 'w') as f:
            self._indices = json.load(f)

    @classmethod
    def fromlist(cls, values, unk=False):
        vocab = cls()
        vocab._values = list(sorted(set(values)))
        vocab._indices = dict((value, i) for i, value in enumerate(vocab))
        vocab._counts = Counter(values)
        vocab.unk = unk
        return vocab

    @property
    def size(self):
        return len(self._values)

    @property
    def unks(self):
        return [value for value in self._values if value.startswith(UNK)]
