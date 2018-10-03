from copy import deepcopy
from typing import NamedTuple
import tempfile

import numpy as np
import torch
import torch.nn as nn
from nltk import Tree

from datatypes import Token, Item, Word, Nonterminal, Action
from actions import SHIFT, REDUCE, NT, GEN
from parser import DiscParser
from model import DiscRNNG, GenRNNG
from eval import evalb
from scripts.get_oracle import unkify, get_actions


class Decoder:
    """Decoder base class for prediction with RNNG."""
    def __init__(self,
                 model=None,
                 dictionary=None,
                 use_chars=False,
                 use_tokenizer=False,
                 verbose=False):
        self.model = model
        self.dictionary = dictionary

        self.use_chars = use_chars  #  using character based input embedding
        self.use_tokenizer = use_tokenizer
        self.verbose = verbose

        self.softmax = nn.Softmax(dim=0)
        self.logsoftmax = nn.LogSoftmax(dim=0)

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
                str, List[str], List[Word].
        """

        pass

    def _init_tokenizer(self):
        # from spacy.tokenizer import Tokenizer
        # nlp = spacy.load('en')
        # self.tokenizer = Tokenizer(nlp.vocab)
        if self.verbose: print("Using NLTK's tokenizer.")
        from nltk import word_tokenize
        self.tokenizer = word_tokenize

    def _tokenize(self, sentence):
        if self.verbose: print('Tokenizing sentence...')
        # return [token.text for token in self.tokenizer(sentence)]
        return [token for token in self.tokenizer(sentence)]

    def _process_unks(self, sentence):
        if self.verbose: print(f'input: {sentence}')
        processed = []
        for word in sentence:
            try:
                self.dictionary.w2i[word]
                processed.append(word)
            except KeyError:
                unk = unkify([word], self.dictionary.w2i)[0]
                processed.append(unk)
        if self.verbose: print(f'unked: {processed}')
        return processed

    def _from_string(self, sentence):
        sentence = self._tokenize(sentence) if self.use_tokenizer else sentence.split()
        if self.verbose: print(f'> {" ".join(sentence)}')
        if not self.use_chars:  # character embedding has no unk token
            processed = self._process_unks(sentence)
        else:
            processed = sentence
        sentence = [Token(orig, proc) for orig, proc in zip(sentence, processed)]
        sentence_items = []
        for token in sentence:
            if self.use_chars:
                index = [self.dictionary.w2i[char] for char in token.processed]
            else:
                index = self.dictionary.w2i[token.processed]
            sentence_items.append(Word(token, index))
        return sentence_items

    def _process_sentence(self, sentence):
        assert len(sentence) > 0, f'decoder received empty sentence'
        if isinstance(sentence, str):
            return self._from_string(sentence)
        elif isinstance(sentence, list):
            if all(isinstance(word, str) for word in sentence):
                return self._from_string(' '.join(sentence))
            if all(isinstance(word, Word) for word in sentence):
                return sentence
        else:
            raise ValueError(f'sentence format not recognized: {sentence}')

    def _compute_probs(self, logits, mask=None, alpha=1.0):
        probs = self.softmax(logits)  # Compute probs.
        if alpha != 1.0:
            probs = probs.pow(alpha)  # Apply temperature scaling.
        if mask is not None:
            assert (mask.shape == probs.shape), mask.shape
            probs = mask * probs
        probs /= probs.sum()  # Renormalize.
        return probs

    def _valid_actions_mask(self):
        mask = torch.Tensor(
            [self.model.is_valid_action(self._make_action(i)) for i in range(3)])
        return mask

    def _best_valid_action(self, logits):
        mask = self._valid_actions_mask()
        masked_logits = torch.Tensor(
            [logit if allowed else -np.inf for logit, allowed in zip(logits, mask)])
        masked_logits, ids = masked_logits.sort(descending=True)
        index = ids[0]
        action = self._make_action(index)
        return index, action

    def load_model(self, path):
        print(f'Loading model from {path}...')
        with open(path, 'rb') as f:
            state = torch.load(f)
        epoch, fscore = state['epochs'], state['test-fscore']
        print(f'Loaded model trained for {epoch} epochs with test-fscore {fscore}.')
        self.model = state['model']
        self.dictionary = state['dictionary']
        self.use_chars = state['args'].use_chars
        self.model.eval()  # Disable dropout.

    def get_tree(self):
        assert len(self.model.stack._items) > 1, 'no tree built yet'
        return self.model.stack._items[1]  # Root node.

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

    def _make_action(self, index):
        """Maps index to action."""
        assert index in range(3), f'invalid action index {index}'
        if index == SHIFT.index:
            return SHIFT
        elif index == REDUCE.index:
            return REDUCE
        elif index == Action.NT_INDEX:
            return NT(Nonterminal('_', -1))  # Content doesn't matter in this case, only type.


class GenerativeDecoder(Decoder):
    """Decoder for generative RNNG by importance sampling."""
    def __init__(self,
                 model=None,
                 proposal=None,
                 dictionary=None,
                 num_samples=100,
                 alpha=0.8,
                 use_chars=False,
                 use_tokenizer=False,
                 verbose=False):
        super(GenerativeDecoder, self).__init__(
            model,
            dictionary,
            use_chars,
            use_tokenizer,
            verbose
        )
        self.proposal = SamplingDecoder(
            model=proposal, dictionary=dictionary, use_chars=use_chars)
        self.num_samples = num_samples
        self.alpha = alpha
        self.i = 0  # current sample index

    def __call__(self, sentence):
        return self.map_tree(sentence)

    def map_tree(self, sentence):
        """Estimate the MAP tree."""
        sentence = self._process_sentence(sentence)
        scored = self.scored_samples(sentence)
        ranked = sorted(scored, reverse=True, key=lambda t: t[-1])
        best_tree, proposal_logprob, logprob = ranked[0]
        return best_tree, proposal_logprob, logprob

    def prob(self, sentence):
        """Estimate the probability of the sentence."""
        sentence = self._process_sentence(sentence)
        prob = torch.zeros(1)
        scored = self.scored_samples(sentence)
        # TODO make this log-sum-exp.
        for _, marginal_logprob, joint_logprob in scored:
            prob += (joint_logprob - marginal_logprob).exp()
        prob /= len(scored)
        return prob

    def scored_samples(self, sentence):
        """Return a list of proposal samples that are scored by the model."""
        if self.use_samples:
            # Retrieve the samples that we've loaded.
            samples = self.samples[self.i]
            self.i += 1
        else:
            # Sample with the proposal model that we've loaded.
            samples = [self.sample(sentence) for _ in range(self.num_samples)]
        scored = [(tree, proposal_logprob, self.score(sentence, tree))
            for tree, proposal_logprob in samples]
        return scored

    def score(self, sentence, tree):
        """Compute p(x,y) under the generative model."""
        self.model.eval()
        words = (word.token.processed for word in sentence)
        actions = self.get_gen_oracle(tree, words)
        self.model.initialize(sentence)
        logprob = 0.0
        for i, action in enumerate(actions):
            # Compute loss
            x = self.model.get_input()
            action_logits = self.model.action_mlp(x).squeeze(0)
            logprob += self.logsoftmax(action_logits)[action.action_index]
            # If we open a nonterminal, predict which.
            if action.is_nt:
                nonterminal_logits = self.model.nonterminal_mlp(x).squeeze(0)
                nt = action.get_nt()
                logprob += self.logsoftmax(nonterminal_logits)[nt.index]
            # If we generate a word, predict which.
            if action.is_gen:
                terminal_logits = self.model.terminal_mlp(x).squeeze(0)
                word = action.get_word()
                logprob += self.logsoftmax(terminal_logits)[word.index]
            self.model.parse_step(action)
        return logprob

    def sample(self, sentence):
        tree, logprob, _ = self.proposal(sentence, alpha=self.alpha)
        return tree, logprob

    def get_gen_oracle(self, tree, sentence):
        actions = []
        tree = tree if isinstance(tree, str) else tree.linearize()
        for a in get_actions(tree):
            if a == 'SHIFT':
                token = next(sentence)
                action = GEN(Word(token, self.dictionary.w2i[token]))
            elif a == 'REDUCE':
                action = Action('REDUCE', Action.REDUCE_INDEX)
            elif a.startswith('NT'):
                nt = a[3:-1]
                action = NT(Nonterminal(nt, self.dictionary.n2i[nt]))
            actions.append(action)
        return actions

    def _make_action(self, index):
        """Maps index to action."""
        assert index in range(3), f'invalid action index {index}'
        if index == Action.GEN_INDEX:
            return GEN(Word('_', -1))  # Content doesn't matter in this case, only type.
        elif index == REDUCE.index:
            return REDUCE
        elif index == Action.NT_INDEX:
            return NT(Nonterminal('_', -1))  # Content doesn't matter in this case, only type.

    def load_model(self, path):
        """Load the (generative) model."""
        super(GenerativeDecoder, self).load_model(path)
        assert isinstance(self.model, GenRNNG), type(self.model)

    def load_proposal_model(self, path):
        """Load the proposal (discriminative) model."""
        print(f'Loading discriminative model (proposal) from `{path}`...')
        with open(path, 'rb') as f:
            state = torch.load(f)
        proposal = state['model']
        assert isinstance(proposal, DiscRNNG), type(proposal)
        epoch, fscore = state['epochs'], state['test-fscore']
        print(f'Loaded discriminative model trained for {epoch} epochs with test-fscore {fscore}.')
        self.proposal = SamplingDecoder(model=proposal, dictionary=dictionary, use_chars=use_chars)
        self.use_samples = False

    def load_proposal_samples(self, path):
        print(f'Loading discriminative samples (proposal) from `{path}`...')
        samples = self._read_samples(path)
        assert all(len(samples[i]) == self.num_samples for i in samples.keys())
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


class GreedyDecoder(DiscriminativeDecoder):
    """Greedy decoder for RNNG."""
    def __call__(self, sentence):
        logprob = 0.0
        num_actions = 0
        self.model.eval()
        sentence = self._process_sentence(sentence)
        with torch.no_grad():
            self.model.initialize(sentence)
            while not self.model.stack.is_empty():
                num_actions += 1
                x = self.model.get_input()
                action_logits = self.model.action_mlp(x).squeeze(0)
                index, action = self._best_valid_action(action_logits)
                logprob += self.logsoftmax(action_logits)[index]
                if action.is_nt:
                    nt_logits = self.model.nonterminal_mlp(x).squeeze(0)
                    nt_logits, ids = nt_logits.sort(descending=True)
                    nt_index = ids[0]
                    nt = self.dictionary.i2n[nt_index]
                    X = Nonterminal(nt, nt_index)
                    action = NT(X)
                    logprob += self.logsoftmax(nt_logits)[0]
                self.model.parse_step(action)
            return self.get_tree(), logprob, num_actions


class SamplingDecoder(DiscriminativeDecoder):
    """Ancestral sampling decoder for RNNG."""
    def __call__(self, sentence, alpha=1.0):
        logprob = 0.0
        num_actions = 0
        self.model.eval()
        with torch.no_grad():
            sentence = self._process_sentence(sentence)
            self.model.initialize(sentence)
            while not self.model.stack.is_empty():
                num_actions += 1
                x = self.model.get_input()
                action_logits = self.model.action_mlp(x).squeeze(0)  # tensor (num_actions)
                mask = self._valid_actions_mask()
                action_probs = self._compute_probs(action_logits, mask=mask, alpha=alpha)
                index = np.random.choice(
                    range(action_probs.size(0)), p=action_probs.data.numpy())
                action = self._make_action(index)
                logprob += self.logsoftmax(action_logits)[index]
                if action.is_nt:
                    nt_logits = self.model.nonterminal_mlp(x).squeeze(0)  # tensor (num_nonterminals)
                    nt_probs = self._compute_probs(nt_logits, alpha=alpha)
                    index = np.random.choice(
                        range(nt_probs.size(0)), p=nt_probs.data.numpy())
                    X = Nonterminal(self.dictionary.i2n[index], index)
                    action = NT(X)
                    logprob += self.logsoftmax(nt_logits)[index]
                self.model.parse_step(action)
            return self.get_tree(), logprob, num_actions


class Beam(NamedTuple):
    parser: DiscParser
    logprob: float


class BeamSearchDecoder(Decoder):
    """Beam search decoder for RNNG."""
    def __call__(self, sentence, k=10):
        with torch.no_grad():
            sentence = self._process_sentence(sentence)
            parser = DiscParser(
                word_embedding=self.model.history.word_embedding,
                nt_embedding=self.model.history.nt_embedding,
                action_embedding=self.model.history.action_embedding,
                stack_encoder=self.model.stack.encoder,
                buffer_encoder=self.model.buffer.encoder,
                history_encoder=self.model.history.encoder,
                device=self.model.device
            )
            # Copy trained empty embedding.
            parser.stack.empty_emb = self.model.stack.empty_emb
            parser.buffer.empty_emb = self.model.buffer.empty_emb
            parser.history.empty_emb = self.model.history.empty_emb
            parser.eval()
            parser.initialize(sentence)
            self.k = k

            self.open_beams = [Beam(parser, 0.0)]
            self.finished = []
            while self.open_beams:
                self.advance_beam()

            finished = [(parser.stack._items[1], logprob) for parser, logprob in self.finished]
            return sorted(finished, key=lambda x: x[1], reverse=True)

    def _best_k_valid_actions(self, parser, logits):
        k = min(self.k, logits.size(0))
        mask = torch.Tensor(
            [parser.is_valid_action(self._make_action(i)) for i in range(3)])
        masked_logits = torch.Tensor(
            [logit if allowed else -np.inf for logit, allowed in zip(logits, mask)])
        masked_logits, ids = masked_logits.sort(descending=True)
        indices = [i.item() for i in ids[:k] if mask[i]]
        return indices, [self._make_action(i) for i in indices]

    def get_input(self, parser):
        stack, buffer, history = parser.get_encoded_input()
        return torch.cat((buffer, history, stack), dim=-1)

    def advance_beam(self):
        """Advance each beam one step and keep best k."""
        new_beams = []
        for beam in self.open_beams:
            parser, log_prob = beam.parser, beam.logprob
            x = self.get_input(parser)
            action_logits = self.model.action_mlp(x).squeeze(0)
            action_logprobs = self.logsoftmax(action_logits)
            indices, best_actions = self._best_k_valid_actions(parser, action_logits)
            for index, action in zip(indices, best_actions):
                new_parser = deepcopy(parser)
                new_log_prob = log_prob + action_logprobs[index]
                if action.is_nt:
                    nt_logits = self.model.nonterminal_mlp(x).squeeze(0)
                    nt_logits, ids = nt_logits.sort(descending=True)
                    nt_logprobs = self.logsoftmax(nt_logits)
                    k = self.k - len(best_actions) + 1  # can open this many Nonterminals.
                    k = min(k, nt_logits.size(0))
                    for i, nt_index in enumerate(ids[:k]):  # nt_logprobs has the same order as ids!
                        new_parser = deepcopy(parser)
                        nt = self.dictionary.i2n[nt_index]
                        X = Nonterminal(nt, nt_index)
                        action = NT(X)
                        new_parser.parse_step(action)
                        new_beams.append(Beam(new_parser, new_log_prob + nt_logprobs[i]))
                else:
                    new_parser.parse_step(action)
                    new_beams.append(Beam(new_parser, new_log_prob))
            del parser
        new_beams = sorted(new_beams, key=lambda x: x[1])[-self.k:]
        self.finished += [beam for beam in new_beams if beam.parser.stack.is_empty()]
        self.open_beams = [beam for beam in new_beams if not beam.parser.stack.is_empty()]


if __name__ == '__main__':
    sentence = u'This is a short sentence but it will do for now .'
    tree = u'(S (NP (DT The) (ADJP (RBS most) (JJ troublesome)) (NN report)) (VP (MD may) ' + \
            '(VP (VB be) (NP (NP (DT the) (NNP August) (NN merchandise) (NN trade) (NN deficit)) ' + \
            '(ADJP (JJ due) (ADVP (IN out)) (NP (NN tomorrow)))))) (. .))'

    greedy = GreedyDecoder()
    greedy.load_model(path='checkpoints/20180815_170655/model.pt')

    beamer = BeamSearchDecoder()
    beamer.load_model(path='checkpoints/20180815_170655/model.pt')

    sampler = SamplingDecoder()
    sampler.load_model(path='checkpoints/20180815_170655/model.pt')

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
