import time
import functools
import math

import dynet as dy
import numpy as np

import trees
import vocabulary


START = '<START>'
STOP = '<STOP>'
UNK = '<UNK>'


class Semiring(object):
    pass


class ProbSemiring(Semiring):

    @staticmethod
    def sum(x, y):
        return x + y

    @staticmethod
    def product(x, y):
        return x * y

    @staticmethod
    def division(x, y):
        return x / y

    @staticmethod
    def as_real(x):
        return x

    @staticmethod
    def sums(xs):
        """Compute the sum over all values."""
        return dy.esum(xs)

    @staticmethod
    def products(values):
        """Compute the product over all values."""
        return dy.constant(1, -np.inf)

    @classmethod
    def zero(cls):
        raise NotImplementedError  # need to find dynet expression for this

    @classmethod
    def one(cls):
        return dy.zeros(1)


class LogProbSemiring(Semiring):

    @staticmethod
    def sum(x, y):
        return dy.logsumexp([x, y])

    @staticmethod
    def product(x, y):
        return x + y

    @staticmethod
    def division(x, y):
        return x - y

    @staticmethod
    def as_real(x):
        return dy.exp(x)

    @staticmethod
    def sums(xs):
        """Compute the sum over all values."""
        return dy.logsumexp(xs)

    @staticmethod
    def products(xs):
        """Compute the product over all values."""
        return dy.esum(xs)

    @staticmethod
    def zero():
        return dy.constant(1, -np.inf)

    @staticmethod
    def one():
        return dy.zeros(1)


class ViterbiSemiring(Semiring):

    @staticmethod
    def sum(x, y):
        return dy.emax([x, y])

    @staticmethod
    def product(x, y):
        return x + y

    @staticmethod
    def division(x, y):
        return x - y

    @staticmethod
    def as_real(x):
        return dy.exp(x)

    @staticmethod
    def sums(xs):
        """Compute the sum over all values."""
        return dy.emax(xs)

    @staticmethod
    def products(xs):
        """Compute the product over all values."""
        return dy.esum(xs)

    @staticmethod
    def zero():
        return dy.constant(1, -np.inf)

    @staticmethod
    def one():
        return dy.zeros(1)


class Feedforward(object):
    def __init__(self, model, input_dim, hidden_dims, output_dim):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Feedforward")

        self.weights = []
        self.biases = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for prev_dim, next_dim in zip(dims, dims[1:]):
            self.weights.append(self.model.add_parameters((next_dim, prev_dim)))
            self.biases.append(self.model.add_parameters(next_dim))

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def __call__(self, x):
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = dy.affine_transform([bias, weight, x])
            if i < len(self.weights) - 1:
                x = dy.rectify(x)
        return x


class ChartParser(object):
    def __init__(
            self,
            model,
            word_vocab,
            label_vocab,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
            span_hidden_dim,
            label_hidden_dim,
            dropout,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Parser")
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim

        self.word_embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))

        self.lstm = dy.BiRNNBuilder(
            lstm_layers,
            word_embedding_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.f_span = Feedforward(
            self.model, 2 * lstm_dim, [span_hidden_dim], 1)
        self.f_label = Feedforward(
            self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size - 1)

        self.dropout = dropout

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def parse(self, words, gold=None):
        is_train = gold is not None

        embeddings = []
        for word in [START] + words + [STOP]:
            # if word not in (START, STOP):
                # count = self.word_vocab.count(word)
                # if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    # word = UNK
            word_embedding = self.word_embeddings[self.word_vocab.index(word)]
            embeddings.append(word_embedding)

        lstm_outputs = self.lstm.transduce(embeddings)

        @functools.lru_cache(maxsize=None)
        def get_span_encoding(left, right):
            forward = (
                lstm_outputs[right][:self.lstm_dim] -
                lstm_outputs[left][:self.lstm_dim])
            backward = (
                lstm_outputs[left + 1][self.lstm_dim:] -
                lstm_outputs[right + 1][self.lstm_dim:])
            return dy.concatenate([forward, backward])

        @functools.lru_cache(maxsize=None)
        def get_span_score(left, right):
            return self.f_span(get_span_encoding(left, right))

        @functools.lru_cache(maxsize=None)
        def get_label_scores(left, right):
            non_empty_label_scores = self.f_label(get_span_encoding(left, right))
            return dy.concatenate([dy.zeros(1), non_empty_label_scores])

        @functools.lru_cache(maxsize=None)
        def recursion(left, right, label, semiring=LogProbSemiring):
            label_score = get_label_scores(left, right)[self.label_vocab.index(label)]
            if right == left + 1:
                return label_score
            else:
                values = []
                for split in range(left+1, right):
                    value = semiring.products([
                        get_span_score(left, split),
                        get_span_score(split, right),
                        semiring.sums([
                            recursion(left, split, l, semiring)
                            for l in self.label_vocab.values]),
                        semiring.sums([
                            recursion(split, right, l, semiring)
                            for l in self.label_vocab.values])
                    ])
                    values.append(value)
                value = semiring.product(label_score, semiring.sums(values))
                return value

        def topsort(n):
            """All nodes in a complete forest over n words in topoligical order."""
            for length in range(1, n + 1):
                for left in range(0, n + 1 - length):
                    right = left + length
                    for label in self.label_vocab.values:
                        yield left, right, label

        def incoming(left, right):
            """Return all incoming nodes to a node that spans left to right."""
            for split in range(left+1, right):
                for left_label in self.label_vocab.values:
                    for right_label in self.label_vocab.values:
                        yield (left, split, right, left_label, right_label)

        def inside(n, semiring=LogProbSemiring):
            chart = dict()
            for node in topsort(n):
                left, right, label = node
                label_score = get_label_scores(left, right)[self.label_vocab.index(label)]

                if right == left + 1:
                    chart[node] = label_score
                else:
                    chart[node] = semiring.zero()
                    for split in range(left+1, right):

                        s_left_split = semiring.sums([
                            chart[left, split, lab] for lab in self.label_vocab.values])

                        s_split_right = semiring.sums([
                            chart[split, right, lab] for lab in self.label_vocab.values])

                        k = semiring.products([
                            label_score,
                            get_span_score(left, split),
                            get_span_score(split, right),
                            s_left_split,
                            s_split_right
                        ])

                        chart[node] = semiring.sum(chart[node], k)
            return chart

        def viterbi(n, tag='*', semiring=ViterbiSemiring):
            chart = dict()
            for node in topsort(n):
                left, right, label = node

                label_score = get_label_scores(left, right)[self.label_vocab.index(label)].value()
                if right == left + 1:
                    children = [trees.LeafParseNode(left, tag, words[left])]
                    subtree = trees.InternalParseNode(label, children)
                    chart[left, right, label] = (label_score, subtree)
                else:
                    subtrees = []
                    for split in range(left+1, right):

                        left_split = get_span_score(left, split).value()
                        right_split = get_span_score(split, right).value()

                        left_score, left_subtree = max(
                            [chart[left, split, lab] for lab in self.label_vocab.values],
                            key=lambda t: t[0])
                        right_score, right_subtree = max(
                            [chart[split, right, lab] for lab in self.label_vocab.values],
                            key=lambda t: t[0])

                        score = label_score + left_split + right_split + left_score + right_score

                        subtrees.append(
                            (score, left_subtree, right_subtree))

                    # for score, left_subtree, right_subtree in sorted(subtrees, key=lambda t: t[0]):
                        # print(round(score,2), label, left, right, '|||', left_subtree.linearize(), '|||', right_subtree.linearize())
                    # print()

                    best_score, best_left_subtree, best_right_subtree = max(subtrees, key=lambda t: t[0])
                    children = [best_left_subtree, best_right_subtree]
                    subtree = trees.InternalParseNode(label, children)
                    chart[left, right, label] = best_score, subtree

            return chart


        if is_train:
            gold_score = LogProbSemiring.products([
                get_label_scores(left, right)[self.label_vocab.index(label)] + \
                get_span_score(left, right)
                for (left, right, label) in gold.spans()
            ])

            I = inside(len(words))
            lognormalizer = I[0, len(words), ('S',)]

            gold_logprob = gold_score - lognormalizer

            # print(len(I))
            # for key, value in I.items():
                # print(key, value.value())
            # print(round(gold_logprob.value(), 2), round(gold_score.value(), 2), round(lognormalizer.value(), 2))
            # print(round(gold_logprob_2.value(), 2), round(gold_score.value(), 2), round(lognormalizer_2.value(), 2))
            # print()

            return -gold_logprob, gold

        else:
            V = viterbi(len(words))
            pred_score, tree = V[0, len(words), ('S',)]

            # print(len(V))
            # for key, value in V.items():
                # score, tree = value
                # print(key, score, tree.linearize())

            return pred_score, tree


def main():

    treebank = trees.load_trees('/Users/daan/data/ptb-benepar/23.auto.clean', strip_top=True)

    tree = treebank[0]
    tree = tree.convert().binarize()  # collapses unaries and gives spans to nodes

    words = [leaf.word for leaf in tree.leaves()]
    print(words)
    print(tree.linearize())

    word_vocab = vocabulary.Vocabulary()
    word_vocab.index(START)
    word_vocab.index(STOP)
    word_vocab.index(UNK)
    for leaf in tree.leaves():
        word_vocab.index(leaf.word)

    label_vocab = vocabulary.Vocabulary()
    label_vocab.index((trees.DUMMY,))
    for label in tree.labels():
        label_vocab.index(label)

    print(label_vocab.values)

    model = dy.ParameterCollection()
    parser = ChartParser(
        model,
        word_vocab,
        label_vocab,
        word_embedding_dim=100,
        lstm_layers=2,
        lstm_dim=128,
        span_hidden_dim=128,
        label_hidden_dim=128,
        dropout=0.4,
    )
    optimizer = dy.AdamTrainer(model, alpha=0.001)

    for i in range(1, 1000):
        dy.renew_cg()

        words = [leaf.word for leaf in tree.leaves()]

        t0 = time.time()
        loss, _ = parser.parse(words, gold=tree)
        t1 = time.time()

        # Update parameters
        loss.forward()
        loss.backward()
        optimizer.update()

        t2 = time.time()

        print(i, round(loss.value(), 2), 'forward', round(t1-t0, 3), 'backward', round(t2-t1, 3))

        if i % 10 == 0:
            viterbi, tree = parser.parse(words)
            print(round(viterbi, 2), tree.unbinarize().convert().linearize())
            print(round(viterbi, 2), tree.convert().linearize())


if __name__ == '__main__':
    main()
