import json
from collections import Counter, defaultdict

import numpy as np


UNK = '<UNK>'


class Vocabulary:
    def __init__(self, unk_value=None):
        self.values = []
        self.indices = {}
        self.counts = defaultdict(int)
        self.unk_value = unk_value
        self.unk_index = 0

    def __iter__(self):
        return iter(self.values)

    @property
    def size(self):
        return len(self.values)

    @classmethod
    def fromlist(cls, values, unk_value=None, sort=True):
        vocab = cls()
        vocab.values = list(sorted(set(values))) if sort else list(values)
        vocab.indices = dict((value, i) for i, value in enumerate(vocab))
        vocab.counts = Counter(values)
        vocab.unk_value = unk_value
        if unk_value is not None:
            vocab.unk_index = vocab.indices[unk_value]
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
        return self.indices[value]

    def count(self, value):
        return self.counts.get(value, 0)

    def index_or_unk(self, value):
        unk_index = self.indices[self.unk_value]  # This also works for the older models
        # return self.indices.get(value, self.unk_index)  # But this is nicer
        return self.indices.get(value, unk_index)

    # Original function, cause of all evil, never to be forgotten.
    # def index_or_unk(self, value):
    #     assert self.unk_value is not None
    #     if value in self.values:
    #         return self.indices[value]
    #     else:
    #         return self.indices[self.unk_value]

    def unkify(self, values):
        assert self.unk_value is not None, 'vocab has no unk'
        unked = []
        for i, word in enumerate(values):
            count = self.count(word)
            if np.random.rand() < 1 / (1 + count):
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
        path = path + '.json' if not path.endswith('.json') else path
        json_dict = {}
        for value, index in self.indices.items():
            count = self.counts[value]
            # NOTE: we use str(value) as a hack to deal with the labels
            # for the crf parser, that is, we turn ('S',) into "('S',)"
            json_dict[str(value)] = dict(index=index, count=count)
        with open(path, 'w') as f:
            json.dump(json_dict, f, indent=4)

    def load(self, path):
        with open(path) as f:
            json_dict = json.load(f)
        self.indices = {}
        self.counts = defaultdict(int)
        for value, value_dict in json_dict.values():
            # NOTE: this turns "('S',)" back into ('S',), see self.save
            if value.startswith("('") and value.endswith(')'):
                value = tuple(value)
            self.indices[value] = value_dict['index']
            self.counts[value] = value_dict['count']

# OLD, before rewriting methods index_or_unk and others
# import json
# from collections import Counter, defaultdict
#
# import numpy as np
#
#
# UNK = '<UNK>'
#
#
# class Vocabulary:
#     def __init__(self, unk=False):
#         self.values = []
#         self.indices = {}
#         self.counts = defaultdict(int)
#         self.unk_value = None
#
#     def __iter__(self):
#         return iter(self.values)
#
#     @property
#     def size(self):
#         return len(self.values)
#
#     @classmethod
#     def fromlist(cls, values, unk_value=None):
#         vocab = cls()
#         vocab.values = list(sorted(set(values)))
#         vocab.indices = dict((value, i) for i, value in enumerate(vocab))
#         vocab.counts = Counter(values)
#         vocab.unk_value = unk_value
#         return vocab
#
#     def add(self, value):
#         self.counts[value] += 1
#         if not value in self.values:
#             self.values.append(value)
#             self.indices[value] = len(self.values) - 1
#
#     def value(self, index):
#         assert 0 <= index < len(self.values)
#         return self.values[index]
#
#     def index(self, value):
#         assert value in self.values, value
#         return self.indices[value]
#
#     def count(self, value):
#         assert value in self.values, value
#         return self.counts[value]
#
#     def index_or_unk(self, value):
#         assert self.unk_value is not None
#         if value in self.values:
#             return self.indices[value]
#         else:
#             return self.indices[self.unk_value]
#
#     def count_or_unk(self, value):
#         assert self.unk_value is not None
#         if value in self.values:
#             return self.counts[value]
#         else:
#             return self.counts[self.unk_value]
#
#     def unkify(self, values):
#         assert self.unk_value is not None
#         unked = []
#         for i, word in enumerate(values):
#             if word not in self.values:
#                 unked.append(self.unk_value)
#             else:
#                 count = self.count(word)
#                 if np.random.rand() < 1 / (1 + count):
#                     unked.append(self.unk_value)
#                 else:
#                     unked.append(word)
#         return unked
#
#     def process(self, values):
#         if self.unk_value is None:
#             return values
#         else:
#             return [self.value(self.index_or_unk(value)) for value in values]
#
#     def save(self, path):
#         path = path + '.json' if not path.endswith('.json') else path
#         json_dict = {}
#         for value, index in self.indices.items():
#             count = self.counts[value]
#             # NOTE: we use str(value) as a hack to deal with the labels
#             # for the crf parser, that is, we turn ('S',) into "('S',)"
#             json_dict[str(value)] = dict(index=index, count=count)
#         with open(path, 'w') as f:
#             json.dump(json_dict, f, indent=4)
#
#     def load(self, path):
#         with open(path) as f:
#             json_dict = json.load(f)
#         self.indices = {}
#         self.counts = defaultdict(int)
#         for value, value_dict in json_dict.values():
#             # NOTE: this turns "('S',)" back into ('S',), see self.save
#             if value.startswith("('") and value.endswith(')'):
#                 value = tuple(value)
#             self.indices[value] = value_dict['index']
#             self.counts[value] = value_dict['count']
