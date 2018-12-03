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
        return dy.cdiv(x, y)

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
        k = dy.ones(1)
        for value in values:
            k = k * value
        return k

    @classmethod
    def zero(cls):
        return dy.zeros(1)

    @classmethod
    def one(cls):
        return dy.ones(1)


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
            # NOTE: made score for empty label trainable
            # self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size - 1)
            self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size)

        self.dropout = dropout

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def parse(self, words, gold=None):
        is_train = gold is not None

        if is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()

        embeddings = []
        for word in [START] + words + [STOP]:
            # NOTE: we skip unking for now since vocab counts are incomplete
            # if word not in (START, STOP):
            #     count = self.word_vocab.count(word)
            #     if not count or (is_train and np.random.rand() < 1 / (1 + count)):
            #         word = UNK
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
            # NOTE: decided to make score for empty label trainable
            # non_empty_label_scores = self.f_label(get_span_encoding(left, right))
            # return dy.concatenate([dy.zeros(1), non_empty_label_scores])
            return self.f_label(get_span_encoding(left, right))

        def topsort(n):
            """All nodes in a complete forest over n words in topoligical order."""
            for length in range(1, n + 1):
                for left in range(0, n + 1 - length):
                    right = left + length
                    for label in self.label_vocab.values:
                        yield left, right, label

        def inside(n, semiring=LogProbSemiring):
            chart = {}
            for node in topsort(n):
                left, right, label = node
                label_index = self.label_vocab.index(label)

                label_score = get_label_scores(left, right)[label_index]
                span_score = get_span_score(left, right)
                edge_weight = semiring.product(label_score, span_score)

                if right == left + 1:
                    chart[node] = edge_weight
                else:
                    chart[node] = semiring.zero()
                    for split in range(left+1, right):

                        sum_left_splits = semiring.sums([
                            chart[left, split, lab] for lab in self.label_vocab.values])

                        sum_right_splits = semiring.sums([
                            chart[split, right, lab] for lab in self.label_vocab.values])

                        k = semiring.products([
                            edge_weight,
                            sum_left_splits,
                            sum_right_splits
                        ])

                        chart[node] = semiring.sum(chart[node], k)

            return chart

        def viterbi(n, tag='*', semiring=ViterbiSemiring):
            chart = {}
            for node in topsort(n):
                left, right, label = node
                label_index = self.label_vocab.index(label)

                label_score = get_label_scores(left, right)[label_index]
                span_score = get_span_score(left, right)
                edge_weight = semiring.product(label_score, span_score)

                if right == left + 1:
                    score = edge_weight
                    children = [trees.LeafParseNode(left, tag, words[left])]
                    subtree = trees.InternalParseNode(label, children)
                    chart[node] = score, subtree
                else:
                    subtrees = []
                    for split in range(left+1, right):

                        left_score, left_subtree = max(
                            [chart[left, split, lab] for lab in self.label_vocab.values],
                            key=lambda t: t[0].value())

                        right_score, right_subtree = max(
                            [chart[split, right, lab] for lab in self.label_vocab.values],
                            key=lambda t: t[0].value())

                        score = semiring.products([
                            edge_weight,
                            left_score,
                            right_score
                        ])

                        subtrees.append(
                            (score, left_subtree, right_subtree))

                    best_score, best_left_subtree, best_right_subtree = max(subtrees, key=lambda t: t[0].value())
                    children = [best_left_subtree, best_right_subtree]
                    subtree = trees.InternalParseNode(label, children)
                    chart[node] = best_score, subtree

            return chart

        semiring = LogProbSemiring

        if is_train:
            # compute the score of the gold tree
            gold_score = semiring.products([
                semiring.product(
                    get_label_scores(left, right)[self.label_vocab.index(label)],
                    get_span_score(left, right))
                for (left, right, label) in gold.spans()
            ])

            inside_chart = inside(len(words))
            # NOTE: maybe do this?
            # lognormalizer = semiring.sums([
            #     inside_chart[0, len(words), lab] for lab in self.label_vocab.values[1:]])  # empty node cannot be top node
            lognormalizer = inside_chart[0, len(words), ('S',)]
            gold_logprob = gold_score - lognormalizer
            loss = -gold_logprob

            # NOTE: alternative max-margin loss
            # correct = gold.unbinarize().convert().linearize() == pred.unbinarize().convert().linearize()
            # loss = dy.zeros(1) if correct else pred_score - gold_score

            # NOTE: some deubugging printing mess follows
            viterbi_chart = viterbi(len(words))
            # NOTE: maybe do this?
            # pred_score, pred = max(
                # [viterbi_chart[0, len(words), lab] for lab in self.label_vocab.values[1:]],  # empty node cannot be top node
                # key=lambda t: t[0].value())
            pred_score, pred = viterbi_chart[0, len(words), ('S',)]

            print('>', gold.unbinarize().convert().linearize())
            print('>', pred.unbinarize().convert().linearize())
            # print('>', self.sample(words).unbinarize().convert().linearize())

            # for key, value in inside_chart.items():
                # print(key, value.value())
            # print('{:.2f} {:.2f} {:.2f}'.format(
                # gold_logprob.value(), gold_score.value(), lognormalizer.value()))

            return loss, gold, inside_chart
        else:
            inside_chart = inside(len(words))
            viterbi_chart = viterbi(len(words))
            lognormalizer = inside_chart[0, len(words), ('S',)]
            pred_score, pred = viterbi_chart[0, len(words), ('S',)]
            pred_logprob = semiring.division(pred_score, lognormalizer)
            return pred_logprob, pred, inside_chart

    def viterbi(self, words, tag='*'):
        """Alternative viterbi that uses inside chart."""
        _, _, chart = self.parse(words)
        chart = {node: np.exp(score.value()) for node, score in chart.items()}

        def recursion(node):
            left, right, label = node
            if right == left + 1:
                children = [trees.LeafParseNode(left, tag, words[left])]
                return trees.InternalParseNode(label, children)
            else:
                splits, scores = [], []
                for split in range(left+1, right):
                    for left_label in self.label_vocab.values:
                        for right_label in self.label_vocab.values:
                            left_child = (left, split, left_label)
                            right_child = (split, right, right_label)
                            score = chart[left_child] * chart[right_child]
                            splits.append((left_child, right_child))
                            scores.append(score)
                left_viterbi_node, right_viterbi_node = splits[np.argmax(scores)]
                children = [recursion(left_viterbi_node), recursion(right_viterbi_node)]
            return trees.InternalParseNode(label, children)

        root = (0, len(words), ('S',))
        return recursion(root)

    def sample(self, words, tag='*'):
        """Sampling using the inside chart."""

        _, _, chart = self.parse(words)
        chart = {node: np.exp(score.value()) for node, score in chart.items()}

        def recursion(node):
            left, right, label = node
            if right == left + 1:
                children = [trees.LeafParseNode(left, tag, words[left])]
                return trees.InternalParseNode(label, children)
            else:
                splits, scores = [], []
                for split in range(left+1, right):
                    for left_label in self.label_vocab.values:
                        for right_label in self.label_vocab.values:
                            left_child = (left, split, left_label)
                            right_child = (split, right, right_label)
                            score = chart[left_child] * chart[right_child]
                            splits.append((left_child, right_child))
                            scores.append(score)
                # sample the children according to score
                probs = np.array(scores) / np.sum(scores)
                i = np.random.choice(range(len(scores)), p=probs)
                left_sampled_node, right_sampled_node = splits[i]
                children = [recursion(left_sampled_node), recursion(right_sampled_node)]
            return trees.InternalParseNode(label, children)

        root = (0, len(words), ('S',))
        return recursion(root)


def main():

    treebank = trees.load_trees(
        '/Users/daan/data/ptb-benepar/23.auto.clean', strip_top=True)

    tree = treebank[0]
    tree = tree.convert().binarize()

    words = [leaf.word for leaf in tree.leaves()]
    print(words)
    print(tree.linearize())
    print(tree.spans())

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
        word_embedding_dim=10,
        lstm_layers=1,
        lstm_dim=10,
        span_hidden_dim=10,
        label_hidden_dim=10,
        dropout=0.,
    )
    optimizer = dy.AdamTrainer(model)

    words = [leaf.word for leaf in tree.leaves()]

    parser.sample(words)

    for i in range(1, 1000):
        dy.renew_cg()

        t0 = time.time()
        loss, _, _ = parser.parse(words, gold=tree)
        t1 = time.time()

        loss.forward()
        loss.backward()
        optimizer.update()

        t2 = time.time()

        print(i, round(loss.value(), 2), round(np.exp(-loss.value()), 2), 'forward', round(t1-t0, 3), 'backward', round(t2-t1, 3))
        print()

if __name__ == '__main__':
    main()
