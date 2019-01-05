import time
import functools
import math

import dynet as dy
import numpy as np
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
        self.word_embeddings = self.model.add_lookup_parameters(
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
                start = 0 if length < len(words) else 1
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
                    # TODO: this is a total hack, and incorrect too
                    chart[left, right, self.label_vocab.values[0]] = semiring.one()
                    for label in self.label_vocab.values[1:]:
                        chart[left, right, label] = semiring.zero()
                        # chart[left, right, label] = semiring.sums([
                        #     inside_chart[left, right, l]
                        #     for l in self.label_vocab.values[1:]
                        #     if l != label
                        # ])
                else:
                    left_split_sum = semiring.zero() if left == 0 else semiring.sums([
                        semiring.product(
                            inside_summed[start, left],
                            outside_summed[start, right])
                        for start in range(0, left)
                    ])
                    right_split_sum = semiring.zero() if right == len(words) else semiring.sums([
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

    def marginals(self, inside_chart, outside_chart, lognormalizer, semiring=LogProbSemiring, eps=1e-4):
        assert set(inside_chart) == set(outside_chart)

        marginals = {}
        for node in inside_chart:
            marginals[node] = dy.exp(
                semiring.division(
                    semiring.product(
                        inside_chart[node],
                        outside_chart[node]),
                    lognormalizer)
            )

        # # TODO: this should not be needed
        # total = dy.esum(list(marginals.values()))
        # if not abs(total.value() - 1.) < eps:
        #     for node, value in marginals.items():
        #         marginals[node] = dy.cdiv(value, total)

        return marginals

    def entropy(self, marginals, label_scores, lognormalizer):

        expected_scores = dy.esum([
            marginals[left, right, label] * label_scores[left, right][self.label_vocab.index(label)]
            for left, right, label in marginals
        ])
        entropy = -expected_scores + lognormalizer

        return entropy

    def forward(self, tree, is_train=True, max_margin=False, return_entropy=True):
        assert isinstance(tree, trees.SpanNode)

        words = tree.words()
        if is_train:
            unked_words = self.word_vocab.unkify(words)
        else:
            unked_words = self.word_vocab.process(words)

        # if max_margin:
        #     label_scores = self.get_node_scores(unked_words)
        #     pred, pred_score = self.viterbi(words, label_scores)
        #     gold_score = self.score_tree(tree, label_scores)
        #     correct = pred.un_cnf().linearize() == tree.un_cnf().linearize()
        #     loss = dy.zeros(1) if correct else pred_score - gold_score
        #     # TODO: this is not working as expected
        #     # print('pred', pred_score.value(), 'gold', gold_score.value())
        #     # print('>', pred.un_cnf().linearize())
        #     # print()
        #     return loss

        label_scores = self.get_node_scores(unked_words)
        score = self.score_tree(tree, label_scores)
        inside_chart, inside_summed, lognormalizer = self.inside(unked_words, label_scores)

        logprob = score - lognormalizer
        nll = -logprob

        if return_entropy:
            # Should not be needed
            # label_scores = {(left, right): dy.inputVector(label_scores[left, right].value())
            #     for left, right in label_scores}
            # inside_summed = {(left, right): dy.scalarInput(inside_summed[left, right].value())
            #     for left, right in inside_summed}
            # lognormalizer = dy.scalarInput(lognormalizer.value())

            outside_chart = self.outside(words, label_scores, inside_chart, inside_summed)
            marginals = self.marginals(inside_chart, outside_chart, lognormalizer)
            entropy = self.entropy(marginals, label_scores, lognormalizer)

            score_sum = dy.esum([
                label_scores[left, right][self.label_vocab.index(label)]
                for left, right, label in marginals
            ])

            print(
                'score-sum', round(score_sum.value(), 2),
                'entropy', round(entropy.value(), 2),
                'marginal-sum', round(dy.esum(list(marginals.values())).value(), 2)
            )
            # return nll, entropy
            return nll
        else:
            return nll

    def parse(self, words):
        unked_words = self.word_vocab.process(words)

        label_scores = self.get_node_scores(unked_words)
        tree, score = self.viterbi(words, label_scores)
        _, _, lognormalizer = self.inside(words, label_scores)

        logprob = score - lognormalizer
        nll = -logprob

        return tree.un_cnf(), nll

    def sample(self, words, num_samples=1):
        semiring = ProbSemiring

        @functools.lru_cache(maxsize=None)
        def get_child_probs(left, right, label):
            splits, scores = [], []
            for split in range(left+1, right):
                for left_label in self.label_vocab.values:
                    for right_label in self.label_vocab.values:
                        left_node = (left, split, left_label)
                        right_node = (split, right, right_label)
                        score = semiring.product(
                            exp_chart_np[left_node], exp_chart_np[right_node])
                        splits.append((left_node, right_node))
                        scores.append(score)
            probs = np.array(scores) / np.sum(scores)
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

        exp_chart_np = {node: np.exp(score.value())
            for node, score in chart_dy.items()}

        top_label_scores = [exp_chart_np[0, len(words), label]
            for label in self.label_vocab.values[1:]]  # dummy label is excluded from top
        top_label_probs = np.array(top_label_scores) / np.sum(top_label_scores)

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

    # def sample(self, words, num_samples=1):
    #     semiring = ProbSemiring
    #
    #     cache = {}
    #     def get_child_probs(left, right, label):
    #         if (left, right, label) in cache:
    #             return cache[left, right, label]
    #         else:
    #             splits, scores = [], []
    #             for split in range(left+1, right):
    #                 for left_label in self.label_vocab.values:
    #                     for right_label in self.label_vocab.values:
    #                         left_node = (left, split, left_label)
    #                         right_node = (split, right, right_label)
    #                         score = semiring.product(
    #                             chart[left_node], chart[right_node])
    #                         splits.append((left_node, right_node))
    #                         scores.append(score)
    #             probs = np.array(scores) / np.sum(scores)
    #             cache[left, right, label] = splits, probs
    #             return splits, probs
    #
    #     def helper(node):
    #         left, right, label = node
    #         if right == left + 1:
    #             children = [trees.LeafSpanNode(left, '*', words[left])]
    #             subtree = trees.InternalSpanNode(label, children)
    #         else:
    #             splits, probs = get_child_probs(left, right, label)
    #             sampled_index = np.random.choice(len(probs), p=probs)
    #             left_sampled_node, right_sampled_node = splits[sampled_index]
    #             left_child = helper(left_sampled_node)
    #             right_child = helper(right_sampled_node)
    #             children = [left_child, right_child]
    #             subtree = trees.InternalSpanNode(label, children)
    #         return subtree
    #
    #     unked_words = self.word_vocab.process(words)
    #
    #     label_scores = self.get_node_scores(unked_words)
    #     chart_dy, lognormalizer = self.inside(words, label_scores)
    #     chart = {node: np.exp(score.npvalue())
    #         for node, score in chart_dy.items()}
    #     label_scores_np = {node: score.npvalue()
    #         for node, score in label_scores.items()}
    #     lognormalizer_np = lognormalizer.npvalue()
    #
    #     top_label_scores = [chart[0, len(words), label]
    #         for label in self.label_vocab.values[1:]]
    #     top_label_probs = np.array(top_label_scores) / np.sum(top_label_scores)
    #
    #     def sample_tree():
    #         # sample top label
    #         sampled_label_index = np.random.choice(
    #             len(top_label_probs), p=top_label_probs) + 1
    #         sampled_label = label_vocab_values[sampled_label_index]
    #         top_node = (0, len(words), sampled_label)
    #
    #         # sample the rest of the tree from that top label
    #         tree = helper(top_node)
    #
    #         # compute the probability of sampled tree
    #         score_np = np.sum([
    #             label_scores_np[left, right, label] for left, right, label in tree.spans()])
    #         logprob = score_np - lognormalizer_np
    #         nll = -logprob
    #
    #         return tree.un_cnf(), nll
    #
    #     samples = Parallel(n_jobs=8)(delayed(sample_tree)() for i in range(num_samples))
    #
    #     if num_samples == 1:
    #         tree, nll = samples.pop()
    #         return tree, nll
    #     else:
    #         return samples

    # def sample(self, words, num_samples=1):
    #     semiring = ProbSemiring
    #
    #     unked_words = self.word_vocab.process(words)
    #     label_scores = self.get_node_scores(unked_words)
    #     chart_dy, lognormalizer = self.inside(words, label_scores)
    #
    #     exp_chart_np = {node: np.exp(score.value())
    #         for node, score in chart_dy.items()}
    #     label_scores_np = {node: score.value()
    #         for node, score in label_scores.items()}
    #     lognormalizer_np = lognormalizer.npvalue()
    #
    #     top_label_scores = [exp_chart_np[0, len(words), label]
    #         for label in self.label_vocab.values[1:]]
    #     top_label_probs = np.array(top_label_scores) / np.sum(top_label_scores)
    #     print(top_label_probs)
    #
    #     def sample_tree(exp_chart_np, label_scores_np, lognormalizer_np, top_label_probs, label_vocab_values):
    #
    #         cache = {}
    #         def get_child_probs(left, right, label):
    #             if (left, right, label) in cache:
    #                 return cache[left, right, label]
    #             else:
    #                 splits, scores = [], []
    #                 for split in range(left+1, right):
    #                     for left_label in label_vocab_values:
    #                         for right_label in label_vocab_values:
    #                             left_node = (left, split, left_label)
    #                             right_node = (split, right, right_label)
    #                             score = semiring.product(
    #                                 exp_chart_np[left_node], exp_chart_np[right_node])
    #                             splits.append((left_node, right_node))
    #                             scores.append(score)
    #                 probs = np.array(scores) / np.sum(scores)
    #                 cache[left, right, label] = splits, probs
    #                 return splits, probs
    #
    #         def helper(node):
    #             left, right, label = node
    #             if right == left + 1:
    #                 children = [trees.LeafSpanNode(left, '*', words[left])]
    #                 subtree = trees.InternalSpanNode(label, children)
    #             else:
    #                 splits, probs = get_child_probs(left, right, label)
    #                 sampled_index = np.random.choice(len(probs), p=probs)
    #                 left_sampled_node, right_sampled_node = splits[sampled_index]
    #                 left_child = helper(left_sampled_node)
    #                 right_child = helper(right_sampled_node)
    #                 children = [left_child, right_child]
    #                 subtree = trees.InternalSpanNode(label, children)
    #             return subtree
    #
    #         # sample top label
    #         sampled_label_index = np.random.choice(
    #             len(top_label_probs), p=top_label_probs) + 1
    #         sampled_label = label_vocab_values[sampled_label_index]
    #         top_node = (0, len(words), sampled_label)
    #
    #         # sample the rest of the tree from that top label
    #         tree = helper(top_node)
    #
    #         # compute the probability of sampled tree
    #         score_np = np.sum([
    #             label_scores_np[left, right, label] for left, right, label in tree.spans()])
    #         logprob = score_np - lognormalizer_np
    #         nll = -logprob
    #
    #         return tree.un_cnf(), nll
    #
    #     samples = Parallel(n_jobs=8)(
    #         delayed(sample_tree)(exp_chart_np,
    #                              label_scores_np,
    #                              lognormalizer_np,
    #                              top_label_probs,
    #                              self.label_vocab.values)
    #         for i in range(num_samples))
    #
    #     if num_samples == 1:
    #         tree, nll = samples.pop()
    #         return tree, nll
    #     else:
    #         return samples
