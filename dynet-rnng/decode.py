import os
import tempfile
from copy import deepcopy
from typing import NamedTuple

import dynet as dy
import numpy as np
from tqdm import tqdm

from actions import SHIFT, REDUCE, NT, GEN
from parser import DiscParser
from model import DiscRNNG, GenRNNG
from tree import Node
from eval import evalb
from data_scripts.get_oracle import unkify, get_actions, get_actions_no_tags
from utils import add_dummy_tags, substitute_leaves, ceil_div


# %%%%%%%%%%%%%%%%%%%%% #
#      Base classes     #
# %%%%%%%%%%%%%%%%%%%%% #

class Decoder:
    """Decoder base class for prediction with RNNG."""
    def __init__(
            self,
            model=None,
            dictionary=None,
            use_tokenizer=False,
            verbose=False
    ):
        self.model = model
        self.dictionary = dictionary
        self.use_tokenizer = use_tokenizer
        self.verbose = verbose
        if self.use_tokenizer:
            self._init_tokenizer()

    def __call__(self, sentence):
        """Decode the sentence with the model.

        This method is different for each deocoder.
        The sentence can be given in various datatypes,
        and will be processed first by `_process_sentence`

        Arguments
        ---------
        sentence: sentence to decode, can be of the following types:
            str, List[str], List[int].
        """
        raise NotImplementedError

    def _init_tokenizer(self):
        from nltk import word_tokenize
        self.tokenizer = word_tokenize

    def _tokenize(self, sentence):
        return [token for token in self.tokenizer(sentence)]

    def _process_unks(self, sentence):
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

    def _compute_probs(self, logits, mask=None, alpha=1.0):
        probs = self.softmax(logits)  # Compute probs.
        if alpha != 1.0:
            probs = probs.pow(alpha)  # Apply temperature scaling.
        if mask is not None:
            assert (mask.shape == probs.shape), mask.shape
            probs = mask * probs
        probs /= probs.sum(dim=-1)  # Renormalize.
        return probs

    def load_model(self, path):
        assert os.path.exists(path), path

        print(f'Loading model from `{path}`...')
        ##
        epoch, fscore = state['epochs'], state['test-fscore']
        self.epoch = epoch
        self.fscore = fscore
        self.model = state['model']
        self.dictionary = state['dictionary']
        ##
        self.model.eval()  # Disable dropout.

    def get_tree(self):
        return self.model.get_tree()

    def from_tree(self, gold):
        """Predicts from a gold tree input and computes fscore with prediction.

        Input should be a unicode string in the :
            u'(S (NP (DT The) (NN equity) (NN market)) (VP (VBD was) (ADJP (JJ illiquid))) (. .))'
        """
        evalb_dir = os.path.expanduser('~/EVALB')  # TODO: this should be part of args.
        # Make a temporay directory for the EVALB files.
        temp_dir = tempfile.TemporaryDirectory(prefix='evalb-')
        gold_path = os.path.join(temp_dir.name, 'gold.txt')
        pred_path = os.path.join(temp_dir.name, 'predicted.txt')
        result_path = os.path.join(temp_dir.name, 'output.txt')
        # Extract sentence from the gold tree.
        sent = Tree.fromstring(gold).leaves()
        # Predict a tree for the sentence.
        pred, *rest = self(sent)
        pred = pred.linearize()
        # Dump these in the temp-file.
        with open(gold_path, 'w') as f:
            print(gold, file=f)
        with open(pred_path, 'w') as f:
            print(pred, file=f)
        fscore = evalb(evalb_dir, pred_path, gold_path, result_path)
        # Cleanup the temporary directory.
        temp_dir.cleanup()
        return pred, fscore


class DiscriminativeDecoder(Decoder):
    """Decoder for discriminative RNNG."""

    def load_model(self, path):
        """Load the discriminative model."""
        super(DiscriminativeDecoder, self).load_model(path)
        assert isinstance(self.model, DiscRNNG), f'must be discriminative model, got `{type(self.model)}`.'
        print(f'Loaded discriminative model trained for {epoch} epochs with test-fscore {fscore}.')


class GenerativeDecoder(Decoder):
    """Decoder for generative RNNG."""

    def load_model(self, path):
        """Load the (generative) model."""
        super(GenerativeDecoder, self).load_model(path)
        assert isinstance(self.model, GenRNNG), f'must be generative model, got `{type(self.model)}`.'
        print(f'Loaded generative model trained for {epoch} epochs with test-fscore {fscore}.')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% #
#   Discriminative decoders   #
# %%%%%%%%%%%%%%%%%%%%%%%%%%% #

class GreedyDecoder(DiscriminativeDecoder):
    """Greedy decoder for discriminative RNNG."""
    def __call__(self, words):
        return self.model.parse(words)


class SamplingDecoder(DiscriminativeDecoder):
    """Ancestral sampling decoder for discriminative RNNG."""
    def __call__(self, words, alpha=1.0):
        return self.model.sample(words, alpha=alpha)


class Beam(NamedTuple):
    parser: DiscParser
    logprob: float


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


# %%%%%%%%%%%%%%%%%%%%%%% #
#   Generative decoders   #
# %%%%%%%%%%%%%%%%%%%%%%% #

class GenerativeSamplingDecoder(GenerativeDecoder):
    """Ancestral sampling decoder for generative RNNG."""
    def __call__(self, alpha=1.0):
        """Returns a sample (x,y) from the model."""
        self.model.sample(alpha=alpha)


class GenerativeImportanceDecoder(GenerativeDecoder):
    """Decoder for generative RNNG by importance sampling."""
    def __init__(
            self,
            model=None,
            dictionary=None,
            num_samples=100,
            alpha=0.8,
            use_tokenizer=False,
            verbose=False
    ):
        super(GenerativeDecoder, self).__init__(
            model,
            dictionary,
            use_tokenizer,
            verbose
        )
        self.num_samples = num_samples
        self.alpha = alpha
        self.i = 0  # current proposal sample index

    def __call__(self, sentence):
        """Return the estimated MAP tree for the sentence."""
        return self.map_tree(sentence)

    def map_tree(self, sentence):
        """Estimate the MAP tree."""
        scored = self.scored_samples(sentence, remove_duplicates=True)  # do not need duplicates for MAP tree
        ranked = sorted(scored, reverse=True, key=lambda t: t[-1])
        best_tree, proposal_logprob, logprob = ranked[0]
        return best_tree, proposal_logprob, logprob

    def logprob(self, sentence):
        """Estimate the probability of the sentence."""
        scored = self.scored_samples(sentence, remove_duplicates=False)  # do need duplicates for perplexity
        logprobs = np.zeros(self.num_samples)
        for i, (tree, marginal_logprob, joint_logprob) in enumerate(scored):
            logprobs[i] = joint_logprob - marginal_logprob
        a = logprobs.max()
        logprob = a + (logprobs - a).exp().mean().log()
        return logprob

    def perplexity(self, sentence):
        return np.exp(-self.logprob(sentence) / len(sentence))

    def scored_samples(self, sentence, remove_duplicates=False):
        sentence = self._process_sentence(sentence)
        return self._scored_samples(sentence, remove_duplicates)

    def _scored_samples(self, words, remove_duplicates):
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
            samples = [self._sample_proposal(words) for _ in range(self.num_samples)]
        # Remove duplicates if we are only interested in reranking.
        if remove_duplicates:
            samples = filter(samples)
        # Score the samples.
        ## Try to make use of autobatch (not working yet)
        dy.renew_cg()
        scores = [self.score(words, tree) for tree, _ in samples]
        dy.esum(scores).value()
        scores = [score.value() for score in scores]
        ##
        # Add dummy tags if trees were loaded from file.
        if self.use_samples:
            scored = [(add_dummy_tags(tree), proposal_logprob, logprob)
                for (tree, proposal_logprob), logprob in zip(samples, scores)]
        else:
            scored = [(tree, proposal_logprob, logprob)
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

    def _sample_proposal(self, words):
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

    def load_proposal_model(self, path):
        """Load the proposal (discriminative) model to sample from."""
        assert os.path.exists(path), path

        self.proposal.load_model(path)
        self.use_samples = False

    def load_proposal_samples(self, path):
        """Load samples from the proposal models."""
        assert os.path.exists(path), path

        print(f'Loading discriminative (proposal) samples from `{path}`...')
        samples = self._read_samples(path)
        assert all(len(samples[i]) == self.num_samples for i in samples.keys()), 'not enough samples'
        self.samples = samples
        self.use_samples = True

    def _read_samples(self, path):
        with open(path) as f:
            lines = [line.strip() for line in f.readlines()]
        idx = 0
        samples = []
        idx2samples = dict()
        for line in lines:
            line_idx, logprob, tree = line.split('|||')
            line_idx, logprob, tree = int(line_idx), float(logprob), tree.strip()
            if line_idx > idx:
                idx2samples[idx] = samples
                idx = line_idx
                samples = []
            samples.append((tree, logprob))
        idx2samples[line_idx] = samples
        return idx2samples


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        exit('Specify model checkpoint to load.')
    else:
        checkpoint = sys.argv[1]

    # A demonstration.
    sentence = u'This is a short sentence but it will do for now .'
    tree = u'(S (NP (DT The) (ADJP (RBS most) (JJ troublesome)) (NN report)) (VP (MD may) ' + \
            '(VP (VB be) (NP (NP (DT the) (NNP August) (NN merchandise) (NN trade) (NN deficit)) ' + \
            '(ADJP (JJ due) (ADVP (IN out)) (NP (NN tomorrow)))))) (. .))'

    greedy = GreedyDecoder()
    greedy.load_model(path=checkpoint)

    beamer = BeamSearchDecoder()
    beamer.load_model(path=checkpoint)

    sampler = SamplingDecoder()
    sampler.load_model(path=checkpoint)

    print('Greedy decoder:')
    tree, logprob, num_actions = greedy(sentence)
    print('{} {:.2f} {:.4f} {}'.format(tree.linearize(with_tag=False), logprob, np.exp(logprob), num_actions))
    print()

    print('Beam-search decoder:')
    results = beamer(sentence, k=2)
    for tree, logprob in results:
        print('{} {:.2f} {:.4f}'.format(tree.linearize(with_tag=False), logprob, np.exp(logprob)))
    print()

    print('Sampling decoder:')
    for _ in range(3):
        tree, logprob, num_actions = sampler(sentence)
        print('{} {:.2f} {:.4f} {}'.format(tree.linearize(with_tag=False), logprob, np.exp(logprob), num_actions))
    print('-'*79)
    print()
