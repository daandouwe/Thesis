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
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    word = UNK
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
            # return dy.emax([self.f_span(get_span_encoding(left, right)), dy.constant(1, -5)])
            return self.f_span(get_span_encoding(left, right))

        @functools.lru_cache(maxsize=None)
        def get_label_scores(left, right):
            # NOTE: decided to make score for empty label trainable
            # non_empty_label_scores = self.f_label(get_span_encoding(left, right))
            # return dy.concatenate([dy.zeros(1), non_empty_label_scores])
            return self.f_label(get_span_encoding(left, right))

        # def topsort(n):
        #     """All nodes in a complete forest over n words in topoligical order."""
        #     for length in range(1, n + 1):
        #         for left in range(0, n + 1 - length):
        #             right = left + length
        #             for label in self.label_vocab.values:
        #                 yield left, right, label

        # def inside_(n, semiring=LogProbSemiring):
        #     chart = {}
        #     edges = {}
        #     for node in topsort(n):
        #         left, right, label = node
        #         label_index = self.label_vocab.index(label)
        #
        #         label_score = get_label_scores(left, right)[label_index]
        #         span_score = get_span_score(left, right)
        #         edge_weight = semiring.product(label_score, span_score)
        #         edges[node] = edge_weight
        #
        #         if right == left + 1:
        #             chart[node] = edge_weight
        #         else:
        #             chart[node] = semiring.zero()
        #             for split in range(left + 1, right):
        #
        #                 sum_left_splits = semiring.sums([
        #                     chart[left, split, lab] for lab in self.label_vocab.values])
        #
        #                 sum_right_splits = semiring.sums([
        #                     chart[split, right, lab] for lab in self.label_vocab.values])
        #
        #                 k = semiring.products([
        #                     edge_weight,
        #                     sum_left_splits,
        #                     sum_right_splits
        #                 ])
        #
        #                 chart[node] = semiring.sum(chart[node], k)
        #
        #     return chart, edges

        # def viterbi_(n, tag='*', semiring=ViterbiSemiring):
        #     chart = {}
        #     for node in topsort(n):
        #         left, right, label = node
        #         label_index = self.label_vocab.index(label)
        #
        #         label_score = get_label_scores(left, right)[label_index]
        #         span_score = get_span_score(left, right)
        #         edge_weight = semiring.product(label_score, span_score)
        #
        #         if right == left + 1:
        #             score = edge_weight
        #             children = [trees.LeafParseNode(left, tag, words[left])]
        #             subtree = trees.InternalParseNode(label, children)
        #             chart[node] = score, subtree
        #         else:
        #             subtrees = []
        #             for split in range(left+1, right):
        #
        #                 left_score, left_subtree = max(
        #                     [chart[left, split, lab] for lab in self.label_vocab.values],
        #                     key=lambda t: t[0].value())
        #
        #                 right_score, right_subtree = max(
        #                     [chart[split, right, lab] for lab in self.label_vocab.values],
        #                     key=lambda t: t[0].value())
        #
        #                 score = semiring.products([
        #                     edge_weight,
        #                     left_score,
        #                     right_score
        #                 ])
        #
        #                 subtrees.append(
        #                     (score, left_subtree, right_subtree))
        #
        #             best_score, best_left_subtree, best_right_subtree = max(subtrees, key=lambda t: t[0].value())
        #             children = [best_left_subtree, best_right_subtree]
        #             subtree = trees.InternalParseNode(label, children)
        #             chart[node] = best_score, subtree
        #
        #     return chart

        # def inside(n, semiring=LogProbSemiring):
        #     chart = {}
        #     for length in range(1, n + 1):
        #         for left in range(0, n + 1 - length):
        #             right = left + length
        #
        #             label_scores = get_label_scores(left, right)
        #             span_score = get_span_score(left, right)
        #
        #             if right == left + 1:
        #                 split_sum = semiring.one()
        #             else:
        #                 split_sum = semiring.sums([
        #                     semiring.product(
        #                         chart[left, split],
        #                         chart[split, right])
        #                     for split in range(left + 1, right)
        #                 ])
        #
        #             # Sum over all labels
        #             start = 0 if length < len(words) else 1  # Do not sum the impossible dummy node
        #             label_sum = semiring.sums([
        #                 label_scores[label_index]
        #                 for label_index in range(start, len(self.label_vocab.values))
        #             ])
        #
        #             score = semiring.products([
        #                 span_score,
        #                 label_sum,
        #                 split_sum
        #             ])
        #
        #             chart[left, right] = score
        #
        #     return chart, None

        def inside(n, semiring=LogProbSemiring):
            chart = {}
            summed = {}
            for length in range(1, n + 1):
                for left in range(0, n + 1 - length):
                    right = left + length

                    label_scores = get_label_scores(left, right)
                    span_score = get_span_score(left, right)

                    if right == left + 1:
                        split_sum = semiring.one()
                    else:
                        split_sum = semiring.sums([
                            semiring.product(
                                summed[left, split],
                                summed[split, right])
                            for split in range(left + 1, right)
                        ])

                    for label, label_index in self.label_vocab.indices.items():
                        chart[left, right, label] = semiring.products([
                            span_score,
                            label_scores[label_index],
                            split_sum
                        ])

                    summed[left, right] = semiring.sums([
                        chart[left, right, label]
                        for label in self.label_vocab.values
                    ])
            normalizer = summed[0, n]

            return chart, normalizer

        def viterbi(n, semiring=LogProbSemiring, tag='*'):
            chart = {}
            for length in range(1, n + 1):
                for left in range(0, n + 1 - length):
                    right = left + length

                    # Obtain scores
                    span_score = get_span_score(left, right)
                    label_scores = get_label_scores(left, right)

                    # Determine best label
                    label_scores_np = label_scores.npvalue()
                    label_index = int(
                        label_scores_np.argmax() if length < len(words) else
                        label_scores_np[1:].argmax() + 1)  # cannot choose dummy node as top node
                    label = self.label_vocab.value(label_index)
                    best_label_score = label_scores[label_index]

                    # Determine the best split point
                    if right == left + 1:
                        best_split_score = semiring.one()
                        children = [trees.LeafParseNode(left, tag, words[left])]
                        subtree = trees.InternalParseNode(label, children)
                    else:
                        best_split = max(
                            range(left + 1, right),
                            key=lambda split:
                                chart[left, split][0].value() +
                                chart[split, right][0].value())
                        best_split_score = semiring.product(
                            chart[left, best_split][0],
                            chart[best_split, right][0]
                        )

                        best_left_subtree = chart[left, best_split][1]
                        best_right_subtree = chart[best_split, right][1]
                        children = [best_left_subtree, best_right_subtree]
                        subtree = trees.InternalParseNode(label, children)

                    score = semiring.products([
                        span_score,
                        best_label_score,
                        best_split_score
                    ])

                    chart[left, right] = score, subtree

            return chart, None

        semiring = LogProbSemiring

        if is_train:

            # compute the score of the gold tree
            gold_score = semiring.products([
                semiring.product(
                    get_label_scores(left, right)[self.label_vocab.index(label)],
                    get_span_score(left, right))
                for left, right, label in gold.spans()
            ])

            inside_chart, normalizer = inside(len(words))
            # normalizer = inside_chart[0, len(words)]
            gold_logprob = semiring.division(gold_score, normalizer)
            loss = -gold_logprob

            viterbi_chart, _ = viterbi(len(words))
            pred_score, pred = viterbi_chart[0, len(words)]

            print('>', gold.unbinarize().convert().linearize())
            print('>', pred.unbinarize().convert().linearize())

            return loss, gold, inside_chart
        else:
            inside_chart, _ = inside(len(words))
            viterbi_chart = viterbi(len(words))
            normalizer = inside_chart[0, len(words), ('S',)]
            pred_score, pred = viterbi_chart[0, len(words), ('S',)]
            pred_logprob = semiring.division(pred_score, normalizer)
            return pred_logprob, pred, inside_chart

    # def viterbi(self, words, tag='*'):
    #     """Alternative viterbi that uses inside chart."""
    #     _, _, chart = self.parse(words)
    #     chart = {node: np.exp(score.value()) for node, score in chart.items()}
    #
    #     def recursion(node):
    #         left, right, label = node
    #         if right == left + 1:
    #             children = [trees.LeafParseNode(left, tag, words[left])]
    #             return trees.InternalParseNode(label, children)
    #         else:
    #             splits, scores = [], []
    #             for split in range(left+1, right):
    #                 for left_label in self.label_vocab.values:
    #                     for right_label in self.label_vocab.values:
    #                         left_child = (left, split, left_label)
    #                         right_child = (split, right, right_label)
    #                         score = chart[left_child] * chart[right_child]
    #                         splits.append((left_child, right_child))
    #                         scores.append(score)
    #             left_viterbi_node, right_viterbi_node = splits[np.argmax(scores)]
    #             children = [recursion(left_viterbi_node), recursion(right_viterbi_node)]
    #         return trees.InternalParseNode(label, children)
    #
    #     root = (0, len(words), ('S',))
    #     return recursion(root)

    def viterbi(self, words, tag='*'):
        """Alternative viterbi that uses inside chart."""
        _, chart = self.parse(words)
        chart = {node: score.value() for node, score in chart.items()}

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
                            score = chart[left_child] + chart[right_child]
                            splits.append((left_child, right_child))
                            scores.append(score)

                left_viterbi_node, right_viterbi_node = splits[np.argmax(scores)]

                children = [recursion(left_viterbi_node), recursion(right_viterbi_node)]
            return trees.InternalParseNode(label, children)

        root = (0, len(words), ('S',))
        tree, score = recursion(root)
        return

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

    tree = treebank[10].convert().binarize()

    # Obtain the word an label vocabularies
    words = [leaf.word for leaf in tree.leaves()]
    # labels = [label for label in tree.labels()]
    labels = [label for tree in treebank[:100] for label in tree.convert().binarize().labels()]

    word_vocab = vocabulary.Vocabulary()
    word_vocab.index(START)
    word_vocab.index(STOP)
    word_vocab.index(UNK)
    for word in words:
        word_vocab.index(word)

    label_vocab = vocabulary.Vocabulary()
    label_vocab.index((trees.DUMMY,))
    for label in labels:
        label_vocab.index(label)

    word_vocab.freeze()
    label_vocab.freeze()

    print(label_vocab.values)
    print(len(label_vocab.values))

    model = dy.ParameterCollection()
    parser = ChartParser(
        model,
        word_vocab,
        label_vocab,
        word_embedding_dim=100,
        lstm_layers=2,
        lstm_dim=100,
        span_hidden_dim=100,
        label_hidden_dim=100,
        dropout=0.,
    )
    optimizer = dy.AdamTrainer(model)

    for i in range(1000):
        dy.renew_cg()

        t0 = time.time()
        loss, _, inside_chart = parser.parse(words, gold=tree)
        t1 = time.time()

        loss.forward()
        loss.backward()
        optimizer.update()

        t2 = time.time()

        print('step', i, 'loss', round(loss.value(), 2), 'forward-time', round(t1-t0, 3), 'backward-time', round(t2-t1, 3), 'length', len(words), 'nodes', len(inside_chart))
        print()

if __name__ == '__main__':
    main()
