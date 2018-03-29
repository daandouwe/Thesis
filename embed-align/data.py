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
        self.w2i, self.i2w = self.make_dicts(path, max_vocab_size, max_lines)

    def make_dicts(self, path, max_vocab_size, max_lines):
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

class Data:
    """
    Data for a monolingual corpus, constructed using the passed dictionary.
    """
    def __init__(self, path, dictionary, max_lines=None):
        self.data = self.load(path, dictionary, max_lines)
        self.lengths = [len(sent) for sent in self.data]

    def load(self, path, dictionary, max_lines):
        data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if max_lines: # pass if max_lines is not None
                    if i > max_lines:
                        break
                line = line.lower().split()
                indices = [dictionary.w2i[w] for w in line]
                data.append(indices)
        return data

class Corpus:
    """
    A monolingual corpus that contains three datasets: train, development
    (validation) and test.
    """
    def __init__(self, train_path, dev_path, test_path,
                 max_vocab_size=None, max_lines=None):
        # The dictionary is constructed based on the training set.
        self.dictionary = Dictionary(train_path, max_vocab_size, max_lines)
        self.train = Data(train_path, self.dictionary, max_lines)
        self.dev = Data(dev_path, self.dictionary)
        self.test = Data(test_path, self.dictionary)

class ParallelCorpus:
    """
    A parallel corpus that holds an l1 and l2 corpus.
    """
    def __init__(self, l1_train_path, l1_dev_path, l1_test_path,
                 l2_train_path, l2_dev_path, l2_test_path,
                 l1_vocab_size, l2_vocab_size, max_lines,
                 ordered=True):
        self.l1 = Corpus(l1_train_path, l1_dev_path, l1_test_path, l1_vocab_size, max_lines)
        self.l2 = Corpus(l2_train_path, l2_dev_path, l2_test_path, l2_vocab_size, max_lines)
        if ordered:
            self.order()
        print("Loaded parallel corpus with {} lines.".format(len(self.l1.train.data)))

    def order(self):
        """
        Orders the sentences in l1.data according to length.
        Uses the same order for l2.data.
        """
        old_order = zip(range(len(self.l1.train.lengths)), self.l1.train.lengths)
        new_order, _ = zip(*sorted(old_order, key=lambda t: t[1]))
        self.l1.train.data = [self.l1.train.data[i] for i in new_order]
        self.l2.train.data = [self.l2.train.data[i] for i in new_order]

    def batches(self, batch_size):
        """
        Returns an iterator over batches in random order.
        """
        n = len(self.l1.train.data)
        batch_order = list(range(0, n, batch_size))
        np.random.shuffle(batch_order)
        for i in batch_order:
            x = pad(self.l1.train.data[i:i+batch_size])
            y = pad(self.l2.train.data[i:i+batch_size])
            yield x, y

    def dev_batches(self, batch_size):
        """
        Returns an iterator over development batches in original order.
        """
        n = len(self.l1.dev.data)
        for i in range(0, n, batch_size):
            x = pad(self.l1.dev.data[i:i+batch_size])
            y = pad(self.l2.dev.data[i:i+batch_size])
            yield x, y

    def test_batches(self, batch_size):
        """
        Returns an iterator over test batches in original order.
        """
        n = len(self.l1.test.data)
        for i in range(0, n, batch_size):
            x = pad(self.l1.test.data[i:i+batch_size])
            y = pad(self.l2.test.data[i:i+batch_size])
            yield x, y


if __name__ == '__main__':
    e_train_path = 'hansards/train/train.e'
    e_dev_path = 'hansards/dev/dev.e'
    e_test_path = 'hansards/test/test.e'
    f_train_path = 'hansards/train/train.f'
    f_dev_path = 'hansards/dev/dev.f'
    f_test_path = 'hansards/test/test.f'

    parallel = ParallelCorpus(e_train_path, e_dev_path, e_test_path,
                              f_train_path, f_dev_path, f_test_path,
                              l1_vocab_size=10000, l2_vocab_size=10000,
                              max_lines=1000, ordered=False)
    batches = parallel.batches(1)
    for _ in range(1):
        x, y = next(batches)
        print([parallel.l1.dictionary.i2w[i] for i in x.data.numpy()[0]])
        print([parallel.l2.dictionary.i2w[i] for i in y.data.numpy()[0]])
