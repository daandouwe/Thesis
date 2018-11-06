import os
import json
from copy import deepcopy
from typing import NamedTuple

import dynet as dy
import numpy as np
from tqdm import tqdm

from actions import SHIFT, REDUCE, NT, GEN
from data import Dictionary
from parser import DiscParser
from model import DiscRNNG, GenRNNG
from tree import Node
from eval import evalb
from utils import ceil_div, add_dummy_tags, substitute_leaves, unkify, get_actions_no_tags


class Decoder:
    """Decoder base class for prediction with RNNG."""
    def __init__(
            self,
            model=None,
            dictionary=None,
            use_tokenizer=False,
    ):
        self.model = model
        self.dictionary = dictionary
        self.use_tokenizer = use_tokenizer
        if self.use_tokenizer:
            self._init_tokenizer()

    def __call__(self, sentence):
        """Decode the sentence with the model.

        This method is different for each deocoder.
        The sentence can be given in various datatypes,
        and will be processed first by `_process_sentence`

        Arguments
        ---------
        sentence : ``str`` or ``List[str]`` or ``List[int]``
            The sentence to decode which can be of various types.
        """
        raise NotImplementedError

    def _init_tokenizer(self):
        from nltk import word_tokenize
        self.tokenizer = word_tokenize

    def _tokenize(self, sentence):
        return [token for token in self.tokenizer(sentence)]

    def _process_unks(self, sentence):
        assert isinstance(list, sentence)

        processed = []
        for word in sentence:
            try:
                self.dictionary.w2i[word]
                processed.append(word)
            except KeyError:
                unk = unkify([word], self.dictionary.w2i)[0]
                processed.append(unk)
        return processed

    def _from_string(self, sentence):
        assert isinstance(sentence, str), sentence

        sentence = self._tokenize(sentence) if self.use_tokenizer else sentence.split()
        processed = self._process_unks(sentence)
        return [self.dictionary.w2i[word] for word in processed]

    def _process_sentence(self, words):
        assert len(words) > 0, f'decoder received empty words'

        if isinstance(words, str):
            return self._from_string(words)
        elif isinstance(words, list) and all(isinstance(word, str) for word in words):
            return self._from_string(' '.join(words))
        elif isinstance(words, list) and all(isinstance(word, int) for word in words):
            return words
        else:
            raise ValueError(f'sentence format not recognized: {sentence}')

    def load_model(self, dir):
        assert os.path.isdir(dir), dir

        print(f'Loading model from `{dir}`...')
        model_checkpoint_path = os.path.join(dir, 'model')
        state_checkpoint_path = os.path.join(dir, 'state.json')
        dict_checkpoint_path = os.path.join(dir, 'dict.json')

        with open(state_checkpoint_path, 'r') as f:
            state = json.load(f)
        print(f"Loaded model trained for {state['epochs']} epochs with test-fscore {state['test-fscore']}.")
        self.dictionary = Dictionary()
        self.dictionary.load(dict_checkpoint_path)
        [self.model] = dy.load(model_checkpoint_path, dy.ParameterCollection())
        self.model.eval()


class DiscriminativeDecoder(Decoder):
    """Decoder for discriminative RNNG."""

    def load_model(self, path):
        """Load the discriminative model."""
        super(DiscriminativeDecoder, self).load_model(path)
        assert isinstance(self.model, DiscRNNG), f'must be discriminative model, got `{type(self.model)}`.'


class GenerativeDecoder(Decoder):
    """Decoder for generative RNNG."""

    def load_model(self, path):
        """Load the (generative) model."""
        super(DiscriminativeDecoder, self).load_model(path)
        assert isinstance(self.model, GenRNNG), f'must be generative model, got `{type(self.model)}`.'


class GreedyDecoder(DiscriminativeDecoder):
    """Greedy decoder for discriminative RNNG."""
    def __call__(self, words):
        words = self._process_sentence(words)
        tree, nll = self.model.parse(words)
        return tree, -nll.value()


class SamplingDecoder(DiscriminativeDecoder):
    """Ancestral sampling decoder for discriminative RNNG."""
    def __call__(self, words, alpha=1.0):
        words = self._process_sentence(words)
        tree, nll = self.model.sample(words, alpha=alpha)
        return tree, -nll.value()


# class Beam(NamedTuple):
#     parser: DiscParser
#     logprob: float
#
#
# class BeamSearchDecoder(DiscriminativeDecoder):
#     """Beam search decoder for discriminative RNNG."""
#     def __call__(self, sentence, k=10):
#         """"""
#         with torch.no_grad():
#             sentence = self._process_sentence(sentence)
#             # Use a separate parser to manage the different beams
#             # (each beam is a separate continuation of this parser.)
#             parser = DiscParser(
#                 word_embedding=self.model.history.word_embedding,
#                 nt_embedding=self.model.history.nt_embedding,
#                 action_embedding=self.model.history.action_embedding,
#                 stack_encoder=self.model.stack.encoder,
#                 buffer_encoder=self.model.buffer.encoder,
#                 history_encoder=self.model.history.encoder,
#                 device=self.model.device
#             )
#             # Copy trained empty embedding.
#             parser.stack.empty_emb = self.model.stack.empty_emb
#             parser.buffer.empty_emb = self.model.buffer.empty_emb
#             parser.history.empty_emb = self.model.history.empty_emb
#             parser.eval()
#             parser.initialize(sentence)
#             self.k = k
#
#             self.open_beams = [Beam(parser, 0.0)]
#             self.finished = []
#             while self.open_beams:
#                 self.advance_beam()
#
#             finished = [(parser.stack._items[1], logprob) for parser, logprob in self.finished]
#             return sorted(finished, key=lambda x: x[1], reverse=True)
#
#     def _best_k_valid_actions(self, parser, logits):
#         k = min(self.k, logits.size(0))
#         mask = torch.Tensor(
#             [parser.is_valid_action(self._make_action(i)) for i in range(3)])
#         masked_logits = torch.Tensor(
#             [logit if allowed else -np.inf for logit, allowed in zip(logits, mask)])
#         masked_logits, ids = masked_logits.sort(descending=True)
#         indices = [i.item() for i in ids[:k] if mask[i]]
#         return indices, [self._make_action(i) for i in indices]
#
#     def get_input(self, parser):
#         stack, buffer, history = parser.get_encoded_input()
#         return torch.cat((buffer, history, stack), dim=-1)
#
#     def advance_beam(self):
#         """Advance each beam one step and keep best k."""
#         new_beams = []
#         for beam in self.open_beams:
#             parser, log_prob = beam.parser, beam.logprob
#             x = self.get_input(parser)
#             action_logits = self.model.action_mlp(x).squeeze(0)
#             action_logprobs = self.logsoftmax(action_logits)
#             indices, best_actions = self._best_k_valid_actions(parser, action_logits)
#             for index, action in zip(indices, best_actions):
#                 new_parser = deepcopy(parser)
#                 new_log_prob = log_prob + action_logprobs[index]
#                 if action.is_nt:
#                     nt_logits = self.model.nonterminal_mlp(x).squeeze(0)
#                     nt_logits, ids = nt_logits.sort(descending=True)
#                     nt_logprobs = self.logsoftmax(nt_logits)
#                     k = self.k - len(best_actions) + 1  # can open this many Nonterminals.
#                     k = min(k, nt_logits.size(0))
#                     for i, nt_index in enumerate(ids[:k]):  # nt_logprobs has the same order as ids!
#                         new_parser = deepcopy(parser)
#                         nt = self.dictionary.i2n[nt_index]
#                         X = Nonterminal(nt, nt_index)
#                         action = NT(X)
#                         new_parser.parse_step(action)
#                         new_beams.append(Beam(new_parser, new_log_prob + nt_logprobs[i]))
#                 else:
#                     new_parser.parse_step(action)
#                     new_beams.append(Beam(new_parser, new_log_prob))
#             del parser
#         new_beams = sorted(new_beams, key=lambda x: x[1])[-self.k:]
#         self.finished += [beam for beam in new_beams if beam.parser.stack.is_empty()]
#         self.open_beams = [beam for beam in new_beams if not beam.parser.stack.is_empty()]


class GenerativeSamplingDecoder(GenerativeDecoder):
    """Ancestral sampling decoder for generative RNNG."""
    def __call__(self, alpha=1.0):
        """Returns a sample (x,y) from the model."""
        tree, nll = self.model.sample(alpha=alpha)
        return tree, -nll.value()


class GenerativeImportanceDecoder(GenerativeDecoder):
    """Decoder for generative RNNG by importance sampling."""
    def __init__(
            self,
            model=None,
            dictionary=None,
            num_samples=100,
            alpha=0.8,
            use_tokenizer=False,
    ):
        super(GenerativeDecoder, self).__init__(
            model,
            dictionary,
            use_tokenizer,
        )
        self.num_samples = num_samples
        self.alpha = alpha
        self.i = 0  # current proposal sample index

    def __call__(self, sentence):
        """Return the estimated MAP tree for the sentence."""
        return self.map_tree(sentence)

    def map_tree(self, sentence):
        """Estimate the MAP tree."""
        sentence = self._process_sentence(sentence)
        scored = self.scored_samples(sentence, remove_duplicates=True)  # do not need duplicates for MAP tree
        ranked = sorted(scored, reverse=True, key=lambda t: t[-1])
        best_tree, proposal_logprob, logprob = ranked[0]
        return best_tree, proposal_logprob, logprob

    def logprob(self, sentence):
        """Estimate the probability of the sentence."""
        sentence = self._process_sentence(sentence)
        scored = self.scored_samples(sentence, remove_duplicates=False)  # do need duplicates for perplexity
        logprobs = np.zeros(self.num_samples)
        for i, (tree, marginal_logprob, joint_logprob) in enumerate(scored):
            logprobs[i] = joint_logprob - marginal_logprob
        a = logprobs.max()
        logprob = a + (logprobs - a).exp().mean().log()
        return logprob

    def perplexity(self, sentence):
        sentence = self._process_sentence(sentence)
        return np.exp(-self.logprob(sentence) / len(sentence))

    def scored_samples(self, words, remove_duplicates):
        """Return a list of proposal samples that will be scored by the joint model."""
        def filter(samples):
            """Filter out duplicate trees from the samples."""
            output = []
            seen = set()
            for tree, logprob in samples:
                if tree not in seen:
                    output.append((tree, logprob))
                    seen.add(tree)
            return output

        assert isinstance(words, list), words
        assert all(isinstance(word, int) for word in words), words

        if self.use_samples:
            # Retrieve the samples that we've loaded.
            samples = self.samples[self.i]
            self.i += 1
        else:
            # Sample with the proposal model that we've loaded.
            samples = [self._sample_one_proposal(words) for _ in range(self.num_samples)]
        # Remove duplicates if we are only interested in reranking.
        if remove_duplicates:
            samples = filter(samples)
        # Score the samples.
        ##
        # Try to make use of autobatch (not working yet)
        # dy.renew_cg()
        # scores = [self.score(words, tree) for tree, _ in samples]
        # This evokes autobatching
        # dy.esum(scores).value()
        # scores = [score.value() for score in scores]
        ##
        dy.renew_cg()
        scores = [self.score(words, tree).value() for tree, _ in samples]
        if self.use_samples:
            # Proposal trees loaded from file have no tags.
            scored = [(add_dummy_tags(tree), proposal_logprob, logprob)
                for (tree, proposal_logprob), logprob in zip(samples, scores)]
        else:
            # Proposal trees sampled can printed with a dummy tag.
            scored = [(tree.linearize(with_tag=True), proposal_logprob, logprob)
                for (tree, proposal_logprob), logprob in zip(samples, scores)]
        return scored

    def score(self, words, tree):
        """Compute log p(x,y) under the generative model."""
        assert isinstance(words, list), words
        assert all(isinstance(word, int) for word in words)

        tree = tree.linearize(with_tag=False) if isinstance(tree, Node) else tree
        oracle = self._get_gen_oracle(tree, [self.dictionary.i2w[i] for i in words])
        actions = [self.dictionary.a2i[action] for action in oracle]
        return -self.model(None, actions)

    def _sample_one_proposal(self, words):
        dy.renew_cg()
        tree, nll = self.proposal(words, alpha=self.alpha)
        return tree, -nll.value()

    def _get_gen_oracle(self, tree, words):
        """Extract the generative action sequence from the tree and sentence."""
        assert isinstance(tree, str), tree
        assert isinstance(words, list) and all(isinstance(word, str) for word in words), words

        words = iter(words)
        return [GEN(next(words)) if action == SHIFT else action
            for action in get_actions_no_tags(tree)]

    def _read_proposals(self, path):
        with open(path) as f:
            lines = [line.strip() for line in f.readlines()]
        id = 0
        samples = []
        proposals = dict()
        for line in lines:
            line_id, logprob, tree = line.split('|||')
            line_id, logprob, tree = int(line_id), float(logprob), tree.strip()
            if line_id > id:
                proposals[id] = samples
                id = line_id
                samples = []
            samples.append((tree, logprob))
        proposals[line_id] = samples
        return proposals

    def load_proposal_model(self, path):
        """Load the proposal (discriminative) model to sample from."""
        assert os.path.exists(path), path

        self.proposal.load_model(path)
        self.use_samples = False

    def load_proposal_samples(self, path):
        """Load samples from the proposal models."""
        assert os.path.exists(path), path

        print(f'Loading discriminative (proposal) samples from `{path}`...')
        samples = self._read_proposals(path)
        assert all(len(samples[i]) == self.num_samples for i in samples.keys()), 'not enough samples'
        self.samples = samples
        self.use_samples = True
