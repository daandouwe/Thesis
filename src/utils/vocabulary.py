import json
from collections import Counter, defaultdict

import numpy as np


UNK = '<UNK>'


class Vocabulary:
    def __init__(self, unk=False):
        self.values = []
        self.indices = {}
        self.counts = defaultdict(int)
        self.unk_value = None

    def __iter__(self):
        return iter(self.values)

    @property
    def size(self):
        return len(self.values)

    @classmethod
    def fromlist(cls, values, unk_value=None):
        vocab = cls()
        vocab.values = list(sorted(set(values)))
        vocab.indices = dict((value, i) for i, value in enumerate(vocab))
        vocab.counts = Counter(values)
        vocab.unk_value = unk_value
        return vocab

    def add(self, value):
        self.counts[value] += 1
        if not value in self.values:
            self.values.append(value)
            self.indices[value] = len(self.values) - 1

    def value(self, index):
        assert 0 <= index < len(self.values)
        return self.values[index]

    def index(self, value):
        assert value in self.values
        return self.indices[value]

    def count(self, value):
        assert value in self.values
        return self.counts[value]

    def index_or_unk(self, value):
        assert self.unk_value is not None
        if value in self.indices:
            return self.indices[value]
        else:
            return self.indices[self.unk_value]

    def unkify(self, values):
        assert self.unk_value is not None
        unked = []
        for i, word in enumerate(values):
            count = self.count(word)
            if not count or np.random.rand() < 1 / (1 + count):
                unked.append(self.unk_value)
            else:
                unked.append(word)
        return unked

    def process(self, values):
        if self.unk_value is None:
            return values
        else:
            return [self.value(self.index_or_unk(value)) for value in values]

    def save(self, path):
        # TODO: do this for everything
        path = path + '.json' if not path.endswith('.json') else path
        with open(path, 'w') as f:
            json.dump(self.indices, f, indent=4)

    def load(self, path):
        # TODO: do this for everything
        with open(path, 'w') as f:
            self.indices = json.load(f)
