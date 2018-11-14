import os
import json
from copy import deepcopy
from typing import NamedTuple

import dynet as dy
import numpy as np

from parser import DiscParser
from model import DiscRNNG, GenRNNG
from tree import fromstring, add_dummy_tags
from utils import ceil_div


class GenerativeDecoder:
    """Decoder for generative RNNG by importance sampling."""
    def __init__(
            self,
            model=None,
            proposal=None,
            num_samples=100,
            alpha=0.8,
    ):
        if model is not None:
            assert isinstance(model, GenRNNG)
        if proposal is not None:
            assert isinstance(proposal, DiscRNNG)

        self.model = model
        self.proposal = proposal
        self.num_samples = num_samples
        self.alpha = alpha
        self.use_samples = False

    def parse(self, words):
        """Return the estimated MAP tree for the words."""
        return self.map_tree(words)

    def map_tree(self, words):
        """Estimate the MAP tree."""
        scored = self.scored_samples(words, remove_duplicates=True)  # do not need duplicates for MAP tree
        ranked = sorted(scored, reverse=True, key=lambda t: t[-1])
        best_tree, proposal_logprob, joint_logprob = ranked[0]
        return best_tree, proposal_logprob, joint_logprob

    def logprob(self, words):
        """Estimate the probability of the words."""
        scored = self.scored_samples(words, remove_duplicates=False)  # do need duplicates for perplexity
        logprobs = np.zeros(self.num_samples)
        for i, (tree, proposal_logprob, joint_logprob) in enumerate(scored):
            logprobs[i] = joint_logprob - proposal_logprob
        a = logprobs.max()
        logprob = a + np.log(np.mean(np.exp(logprobs - a)))  # log-mean-exp
        return logprob

    def perplexity(self, words):
        return np.exp(-self.logprob(words) / len(list(words)))

    def remove_duplicates(self, samples):
        """Filter out duplicate trees from the samples."""
        output = []
        seen = set()
        for tree, logprob in samples:
            if tree.linearize() not in seen:
                output.append((tree, logprob))
                seen.add(tree.linearize())
        return output

    def scored_samples(self, words, remove_duplicates=False):
        """Return a list of proposal samples that will be scored by the joint model."""
        if self.use_samples:
            samples = next(self.samples)
        else:
            dy.renew_cg()
            words = list(words)
            samples = []
            for _ in range(self.num_samples):
                tree, nll = self.proposal.sample(words, alpha=self.alpha)
                samples.append((tree, -nll.value()))

        if remove_duplicates:
            samples = self.remove_duplicates(samples)
            print(f'{len(samples)}/{self.num_samples} unique')

        # Score the samples.
        joint_logprobs = [-self.model.forward(tree, is_train=False).value() for tree, _ in samples]

        # Merge the two lists.
        scored = [(tree, proposal_logprob, joint_logprob)
            for (tree, proposal_logprob), joint_logprob in zip(samples, joint_logprobs)]

        return scored

    def _read_proposals(self, path):
        print(f'Loading discriminative (proposal) samples from `{path}`...')
        with open(path) as f:
            lines = [line.strip() for line in f.readlines()]
        sent_id = 0
        samples = []
        proposals = []
        for line in lines:
            sample_id, logprob, tree = line.split('|||')
            sample_id, logprob, tree = int(sample_id), float(logprob), fromstring(add_dummy_tags(tree.strip()))
            if sample_id > sent_id:
                # Arrived at the first sample of next sentence
                assert len(samples) == self.num_samples, f'not enough samples for line {sample_id}'
                proposals.append(samples)
                sent_id = sample_id
                samples = []
            samples.append((tree, logprob))
        proposals.append(samples)
        return proposals

    def load_proposal_model(self, dir):
        """Load the proposal model to sample from."""
        assert os.path.isdir(dir), dir

        print(f'Loading proposal model from `{dir}`...')
        model_checkpoint_path = os.path.join(dir, 'model')
        state_checkpoint_path = os.path.join(dir, 'state.json')
        [self.model] = dy.load(model_checkpoint_path, dy.ParameterCollection())
        assert isinstance(proposal, DiscRNNG), f'expected discriminative model got {type(self.model)}'

        with open(state_checkpoint_path, 'r') as f:
            state = json.load(f)
        epochs = state['epochs']
        fscore = state['test-fscore']
        print(f'Loaded model trained for {epochs} epochs with test-fscore {fscore}.')

        self.model.eval()
        self.use_samples = False

    def load_proposal_samples(self, path):
        """Load saved samples from the proposal models."""
        assert os.path.exists(path), path

        self.samples = iter(self._read_proposals(path))
        self.use_samples = True



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
