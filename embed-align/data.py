from collections import Counter, defaultdict

import numpy as np
import torch
from torch.autograd import Variable

PAD_TOKEN = '<pad>'
PAD_INDEX = 0

UNK_TOKEN = '<unk>'
UNK_INDEX = 1

def pad(batch):
    lens = list(map(len, batch))
    max_len = max(lens)
    padded_batch = []
    for k, seq in zip(lens, batch):
        padded =  seq + (max_len - k)*[PAD_INDEX]
        padded_batch.append(padded)
    return Variable(torch.LongTensor(padded_batch))

class Dictionary:

    def __init__(self, path, max_vocab_size, max_lines=None):
        self.w2i, self.i2w = self.make_vocab(path, max_vocab_size, max_lines)

    def make_vocab(self, path, max_vocab_size, max_lines):
        vocab = Counter()
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if max_lines: # pass if max_lines is not None
                    if i > max_lines:
                        break
                line = line.lower().split()
                vocab.update(line)
        w2i, i2w = dict(), dict()
        w2i[PAD_TOKEN] = PAD_INDEX
        w2i[UNK_TOKEN] = UNK_INDEX
        i2w[PAD_INDEX] = PAD_TOKEN
        i2w[UNK_INDEX] = UNK_TOKEN
        n = max_vocab_size if max_vocab_size else len(vocab)
        for i, (w, _) in enumerate(vocab.most_common(n-1), 1):
            w2i[w] = i
            i2w[i] = w
        del vocab
        w2i = defaultdict(lambda : UNK_INDEX, w2i)
        i2w = defaultdict(lambda : UNK_TOKEN, i2w)
        return w2i, i2w

class Corpus:

    def __init__(self, path, max_vocab_size=None, max_lines=None):
        self.dictionary = Dictionary(path, max_vocab_size, max_lines)
        self.data = self.load(path, max_lines)
        self.lengths = [len(sent) for sent in self.data]

    def load(self, path, max_lines):
        data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if max_lines: # pass if max_lines is not None
                    if i > max_lines:
                        break
                line = line.lower().split()
                indices = [self.dictionary.w2i[w] for w in line]
                data.append(indices)
        return data

class ParallelCorpus:
    """
    A parallel corpus that for that holds two languages: l1 and l2.
    """
    def __init__(self, l1_path, l2_path, max_vocab_size, max_lines, ordered=True):
        self.l1 = Corpus(l1_path, max_vocab_size, max_lines)
        self.l2 = Corpus(l2_path, max_vocab_size, max_lines)
        if ordered:
            self.order()

        print("Loaded parallel corpus with {} lines.".format(len(self.l1.data)))

    def order(self):
        """
        Orders the sentences in l1.data according to length.
        Uses the same order for l2.data.
        """
        old_order = zip(range(len(self.l1.lengths)), self.l1.lengths)
        new_order, _ = zip(*sorted(old_order, key=lambda t: t[1]))
        self.l1.data = [self.l1.data[i] for i in new_order]
        self.l2.data = [self.l2.data[i] for i in new_order]

    def batches(self, batch_size):
        """
        Returns an iterator over batches in random order.
        """
        n = len(self.l1.data)
        batch_order = list(range(0, n, batch_size))
        np.random.shuffle(batch_order)
        for i in batch_order:
            x = pad(self.l1.data[i:i+batch_size])
            y = pad(self.l2.data[i:i+batch_size])
            yield x, y


if __name__ == '__main__':
    english_path = 'hansards/hansards.36.2.e'
    french_path = 'hansards/hansards.36.2.f'

    parallel = ParallelCorpus(english_path, french_path, max_vocab_size=10000, max_lines=1000, ordered=False)

    batches = parallel.batches(10)
    for _ in range(3):
        print(next(batches))
