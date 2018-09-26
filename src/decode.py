from copy import deepcopy
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from PYEVALB import parser, scorer

from datatypes import Token, Item, Word, Nonterminal, Action
from actions import SHIFT, REDUCE, NT, GEN
from parser import DiscParser
from scripts.get_oracle import unkify


class Decoder:
    """Decoder base class for discriminative RNNG."""
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
        pass

    def _init_tokenizer(self):
        if self.verbose: print("Using spaCy's Engish tokenizer.")
        # from spacy.tokenizer import Tokenizer
        # nlp = spacy.load('en')
        # self.tokenizer = Tokenizer(nlp.vocab)
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
        assert sentence, f'decoder received empty sentence'
        if isinstance(sentence, str):
            return self._from_string(sentence)
        elif isinstance(sentence, list) and isinstance(sentence[0], str):
            return self._from_string(' '.join(sentence))
        else:
            raise ValueError(f'sentence format not recognized: {sentence}')

    def _compute_probs(self, logits, mask=None, alpha=None):
        probs = self.softmax(logits)  # Compute probs.
        if alpha is not None:
            probs = probs.pow(alpha)  # Apply temperature scaling.
        if mask is not None:
            probs = mask * probs
        probs /= probs.sum()  # Renormalize.
        return probs

    def _make_action(self, index):
        """Maps index to action."""
        assert index in range(3), f'invalid action index {index}'
        if index == SHIFT.index:
            return SHIFT
        elif index == REDUCE.index:
            return REDUCE
        elif index == Action.NT_INDEX:
            return NT(Nonterminal('_', -1))  # Content doesn't matter in this case, only type.

    def _valid_actions_mask(self):
        mask = torch.Tensor(
            [self.model.is_valid_action(self._make_action(i)) for i in range(3)]
        )
        return mask

    def _best_valid_action(self, logits):
        mask = self._valid_actions_mask()
        masked_logits = torch.Tensor(
            [logit if allowed else -np.inf for logit, allowed in zip(logits, mask)]
        )
        masked_logits, ids = masked_logits.sort(descending=True)
        index = ids[0]
        action = self._make_action(index)
        return index, action

    def load_model(self, path):
        print(f'Loading model from {path}...')
        with open(path, 'rb') as f:
            state = torch.load(f)
        epoch, fscore = state['epoch'], state['test-fscore']
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
        evalb = scorer.Scorer()
        try:
            gold_tree = parser.create_from_bracket_string(gold)
        except:
            raise ValueError(f'probably not a proper tree: {gold}')
        sent = gold_tree.sentence
        pred, *rest = self(sent)
        pred = pred.linearize()
        pred_tree = parser.create_from_bracket_string(pred)
        result = evalb.score_trees(gold_tree, pred_tree)
        prec, recall = result.prec, result.recall
        fscore = 2 * (prec * recall) / (prec + recall)
        return pred, fscore


class GreedyDecoder(Decoder):
    """Greedy decoder for RNNG."""
    def __call__(self, sentence):
        logprob = 0.0
        num_actions = 0
        with torch.no_grad():
            self.model.eval()
            sentence = self._process_sentence(sentence)
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


class SamplingDecoder(Decoder):
    """Ancestral sampling decoder for RNNG."""
    def __call__(self, sentence, alpha=1.0):
        logprob = 0.0
        num_actions = 0
        with torch.no_grad():
            self.model.eval()
            sentence = self._process_sentence(sentence)
            self.model.initialize(sentence)
            while not self.model.stack.is_empty():
                num_actions += 1
                x = self.model.get_input()
                action_logits = self.model.action_mlp(x).squeeze(0)  # tensor (num_actions)
                mask = self._valid_actions_mask()
                action_probs = self._compute_probs(action_logits, mask=mask, alpha=alpha)
                index = np.random.choice(
                    range(action_probs.size(0)), p=action_probs.data.numpy()
                )
                action = self._make_action(index)
                logprob += self.logsoftmax(action_logits)[index]
                if action.is_nt:
                    nt_logits = self.model.nonterminal_mlp(x).squeeze(0)  # tensor (num_nonterminals)
                    nt_probs = self._compute_probs(nt_logits, alpha)
                    index = np.random.choice(
                        range(nt_probs.size(0)), p=nt_probs.data.numpy()
                    )
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
            [parser.is_valid_action(self._make_action(i)) for i in range(3)]
        )
        masked_logits = torch.Tensor(
            [logit if allowed else -np.inf for logit, allowed in zip(logits, mask)]
        )
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
