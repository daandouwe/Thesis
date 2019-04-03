import time
import functools
import math

import dynet as dy
import numpy as np
import scipy.special as special
from joblib import Parallel, delayed

import utils.trees as trees
from components.feedforward import Feedforward
from .semirings import LogProbSemiring, ProbSemiring


START = '<START>'
STOP = '<STOP>'


class ChartParser(object):
    def __init__(
            self,
            model,
            word_vocab,
            label_vocab,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
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

        # TODO: use the embedding classes
        self.word_embedding = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))

        self.lstm = dy.BiRNNBuilder(
            lstm_layers,
            word_embedding_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.f_label = Feedforward(
            self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size)

        self.dropout = dropout

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    @property
    def num_params(self):
        return sum(np.prod(p.shape()) for p in self.model.parameters_list())

    def train(self):
        """Enable dropout."""
        self.f_label.train()
        self.lstm.set_dropout(self.dropout)

    def eval(self):
        """Disable dropout."""
        self.f_label.eval()
        self.lstm.disable_dropout()

    def get_node_scores(self, words):

        embeddings = []
        for word in [START] + words + [STOP]:
            word_embedding = self.word_embedding[self.word_vocab.index(word)]
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
        def get_label_scores(left, right):
            return self.f_label(get_span_encoding(left, right))

        label_scores = {}

        for length in range(1, len(words) + 1):
            for left in range(0, len(words) + 1 - length):
                right = left + length
                label_scores[left, right] = get_label_scores(left, right)

        return label_scores

    def score_tree(self, tree, label_scores, semiring=LogProbSemiring):
        tree_score = semiring.products([
            label_scores[left, right][self.label_vocab.index(label)]
            for left, right, label in tree.spans()
        ])
        return tree_score

    def inside(self, words, label_scores, semiring=LogProbSemiring):

        chart = {}
        summed = {}

        for length in range(1, len(words) + 1):
            for left in range(0, len(words) + 1 - length):
                right = left + length

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
                    chart[left, right, label] = semiring.product(
                        label_scores[left, right][label_index],
                        split_sum)

                # the dummy node cannot be the top node
                # start = 0 if length < len(words) else 1
                start = 0

                summed[left, right] = semiring.sums([
                    chart[left, right, label]
                    for label in self.label_vocab.values[start:]
                ])

        # implicit top node that expands to all but the dummy node
        lognormalizer = summed[0, len(words)]

        return chart, summed, lognormalizer

    def outside(self, words, label_scores, inside_chart, inside_summed, semiring=LogProbSemiring):

        chart = {}
        outside_summed = {}

        for length in range(len(words), 0, -1):
            for left in range(len(words) - length, -1, -1):
                right = left + length

                if left == 0 and right == len(words):

                    # All edges from TOP have the same weight:
                    # for label in self.label_vocab.values:
                    #     chart[left, right, label] = semiring.division(
                    #         semiring.one(),
                    #         np.log(len(self.label_vocab.indices)))

                    for label, label_index in self.label_vocab.indices.items():
                        chart[left, right, label] = semiring.one()

                else:
                    if left == 0:
                        left_split_sum = semiring.zero()
                    else:
                        left_split_sum = semiring.sums([
                            semiring.product(
                                inside_summed[start, left],
                                outside_summed[start, right])
                            for start in range(0, left)
                        ])

                    if right == len(words):
                        right_split_sum = semiring.zero()
                    else:
                        right_split_sum = semiring.sums([
                            semiring.product(
                                inside_summed[right, end],
                                outside_summed[left, end])
                            for end in range(right + 1, len(words) + 1)
                        ])

                    split_sum = semiring.sum(left_split_sum, right_split_sum)
                    for label in self.label_vocab.values:
                        chart[left, right, label] = split_sum

                outside_summed[left, right] = semiring.sums([
                    semiring.product(
                        label_scores[left, right][label_index],
                        chart[left, right, label])
                    for label, label_index in self.label_vocab.indices.items()
                ])

        return chart

    def viterbi(self, words, label_scores, semiring=LogProbSemiring, tag='*'):

        chart = {}

        for length in range(1, len(words) + 1):
            for left in range(0, len(words) + 1 - length):
                right = left + length

                # Determine best label
                label_scores_np = label_scores[left, right].npvalue()
                label_index = int(
                    label_scores_np.argmax() if length < len(words) else
                    label_scores_np[1:].argmax() + 1)  # cannot choose dummy node as top node
                label = self.label_vocab.value(label_index)
                label_score = label_scores[left, right][label_index]

                # Determine the best split point
                if right == left + 1:
                    best_split_score = semiring.one()
                    children = [trees.LeafSpanNode(left, tag, words[left])]
                    subtree = trees.InternalSpanNode(label, children)
                else:
                    best_split = max(
                        range(left + 1, right),
                        key=lambda split:
                            chart[left, split][1].value() +
                            chart[split, right][1].value())

                    left_subtree, left_score = chart[left, best_split]
                    right_subtree, right_score = chart[best_split, right]

                    children = [left_subtree, right_subtree]
                    subtree = trees.InternalSpanNode(label, children)

                    best_split_score = semiring.product(
                        left_score, right_score)

                score = semiring.product(
                    label_score, best_split_score)

                chart[left, right] = subtree, score

        tree, score = chart[0, len(words)]

        return tree, score

    def marginals(self, inside_chart, outside_chart, lognormalizer, semiring=LogProbSemiring):
        # assert set(inside_chart) == set(outside_chart)  # must contain the same nodes

        marginals = {}
        for node in inside_chart:
            marginals[node] = dy.exp(
                semiring.division(
                    semiring.product(
                        inside_chart[node],
                        outside_chart[node]),
                    lognormalizer))

        return marginals

    def compute_entropy(self, marginals, label_scores, lognormalizer):

        all_scores = [label_scores[left, right][self.label_vocab.index(label)]
            for left, right, label in marginals]

        expected_scores = [
            marginals[left, right, label] * label_scores[left, right][self.label_vocab.index(label)]
            for left, right, label in marginals
        ]

        entropy = lognormalizer - dy.esum(expected_scores)

        return entropy

    def forward(self, tree, is_train=True, return_entropy=False):
        assert isinstance(tree, trees.SpanNode)

        words = tree.words()
        if is_train:
            unked_words = self.word_vocab.unkify(words)
        else:
            unked_words = self.word_vocab.process(words)

        label_scores = self.get_node_scores(unked_words)
        score = self.score_tree(tree, label_scores)
        inside_chart, inside_summed, lognormalizer = self.inside(unked_words, label_scores)
        logprob = score - lognormalizer
        nll = -logprob

        if return_entropy:
            outside_chart = self.outside(words, label_scores, inside_chart, inside_summed)
            marginals = self.marginals(inside_chart, outside_chart, lognormalizer)
            entropy = self.compute_entropy(marginals, label_scores, lognormalizer)
            return nll, entropy
        else:
            return nll

    def entropy(self, words):
        unked_words = self.word_vocab.process(words)
        label_scores = self.get_node_scores(unked_words)
        inside_chart, inside_summed, lognormalizer = self.inside(unked_words, label_scores)
        outside_chart = self.outside(words, label_scores, inside_chart, inside_summed)
        marginals = self.marginals(inside_chart, outside_chart, lognormalizer)
        entropy = self.compute_entropy(marginals, label_scores, lognormalizer)
        return entropy

    def parse(self, words):
        unked_words = self.word_vocab.process(words)

        label_scores = self.get_node_scores(unked_words)
        tree, score = self.viterbi(words, label_scores)
        _, _, lognormalizer = self.inside(words, label_scores)

        logprob = score - lognormalizer
        nll = -logprob

        return tree.un_cnf(), nll

    def sample(self, words, num_samples=1):
        semiring = LogProbSemiring

        @functools.lru_cache(maxsize=None)
        def get_child_probs(left, right, label):
            splits, scores = [], []
            for split in range(left+1, right):
                for left_label in self.label_vocab.values:
                    for right_label in self.label_vocab.values:
                        left_node = (left, split, left_label)
                        right_node = (split, right, right_label)
                        score = semiring.product(
                            chart_np[left_node], chart_np[right_node])
                        splits.append((left_node, right_node))
                        scores.append(score)
            probs = special.softmax(scores)
            return splits, probs

        def helper(node):
            left, right, label = node
            if right == left + 1:
                children = [trees.LeafSpanNode(left, '*', words[left])]
                subtree = trees.InternalSpanNode(label, children)
            else:
                splits, probs = get_child_probs(left, right, label)
                sampled_index = np.random.choice(len(probs), p=probs)
                left_sampled_node, right_sampled_node = splits[sampled_index]
                left_child = helper(left_sampled_node)
                right_child = helper(right_sampled_node)
                children = [left_child, right_child]
                subtree = trees.InternalSpanNode(label, children)
            return subtree

        unked_words = self.word_vocab.process(words)
        label_scores = self.get_node_scores(unked_words)
        chart_dy, _, lognormalizer = self.inside(words, label_scores)

        chart_np = {node: score.value()
            for node, score in chart_dy.items()}

        top_label_logscores = [chart_np[0, len(words), label]
            for label in self.label_vocab.values[1:]]  # dummy label is excluded from top
        top_label_probs = special.softmax(top_label_logscores)

        samples = []
        for _ in range(num_samples):

            # sample top label
            sampled_label_index = np.random.choice(
                len(top_label_probs), p=top_label_probs) + 1
            sampled_label = self.label_vocab.values[sampled_label_index]
            top_label = (0, len(words), sampled_label)

            # sample the rest of the tree from that top label
            tree = helper(top_label)

            # compute the probability of sampled tree
            score = self.score_tree(tree, label_scores)
            logprob = score - lognormalizer
            nll = -logprob

            samples.append((tree.un_cnf(), nll))

        if num_samples == 1:
            tree, nll = samples.pop()
            return tree, nll
        else:
            return samples


    def _sample(self, inside_chart, words, label_scores, lognormalizer, num_samples):
        semiring = LogProbSemiring

        @functools.lru_cache(maxsize=None)
        def get_child_probs(left, right, label):
            splits, scores = [], []
            for split in range(left+1, right):
                for left_label in self.label_vocab.values:
                    for right_label in self.label_vocab.values:
                        left_node = (left, split, left_label)
                        right_node = (split, right, right_label)
                        score = semiring.product(
                            chart_np[left_node], chart_np[right_node])
                        splits.append((left_node, right_node))
                        scores.append(score)
            probs = special.softmax(scores)
            return splits, probs

        def helper(node):
            left, right, label = node
            if right == left + 1:
                children = [trees.LeafSpanNode(left, '*', words[left])]
                subtree = trees.InternalSpanNode(label, children)
            else:
                splits, probs = get_child_probs(left, right, label)
                sampled_index = np.random.choice(len(probs), p=probs)
                left_sampled_node, right_sampled_node = splits[sampled_index]
                left_child = helper(left_sampled_node)
                right_child = helper(right_sampled_node)
                children = [left_child, right_child]
                subtree = trees.InternalSpanNode(label, children)
            return subtree

        chart_np = {node: score.value()
            for node, score in inside_chart.items()}

        top_label_logscores = [chart_np[0, len(words), label]
            for label in self.label_vocab.values[1:]]  # dummy label is excluded from top
        top_label_probs = special.softmax(top_label_logscores)

        samples = []
        for _ in range(num_samples):

            # sample top label
            sampled_label_index = np.random.choice(
                len(top_label_probs), p=top_label_probs) + 1
            sampled_label = self.label_vocab.values[sampled_label_index]
            top_label = (0, len(words), sampled_label)

            # sample the rest of the tree from that top label
            tree = helper(top_label)

            # compute the probability of sampled tree
            score = self.score_tree(tree, label_scores)
            logprob = score - lognormalizer
            nll = -logprob

            samples.append((tree.un_cnf(), nll))

        return samples

    def parse_sample_entropy(self, words, num_samples):
        # shared computation
        unked_words = self.word_vocab.process(words)
        label_scores = self.get_node_scores(unked_words)
        inside_chart, inside_summed, lognormalizer = self.inside(unked_words, label_scores)

        # entropy computation
        outside_chart = self.outside(words, label_scores, inside_chart, inside_summed)
        marginals = self.marginals(inside_chart, outside_chart, lognormalizer)
        entropy = self.compute_entropy(marginals, label_scores, lognormalizer)

        # parse computation
        parse, parse_score = self.viterbi(words, label_scores)
        parse = parse.un_cnf()

        # sample computation
        samples = self._sample(inside_chart, words, label_scores, lognormalizer, num_samples)

        return parse, samples, entropy
