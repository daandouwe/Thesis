from copy import deepcopy
import json

import numpy as np
import torch
import torch.nn as nn

from datatypes import Item, Word, Nonterminal, Action
from actions import SHIFT, REDUCE, NT, GEN
from parser_test import Parser
from scripts.get_oracle import unkify


class Decoder:
    """Decoder base class for discriminative RNNG."""
    def __init__(self,
                 model=None,
                 dictionary=None,
                 use_char=False,
                 use_tokenizer=False,
                 verbose=False):
        self.model = model
        self.dictionary = dictionary

        self.use_char = use_char  # character based input embedding
        self.use_tokenizer = use_tokenizer
        self.verbose = verbose

        self._logprobs = []  # log probabilities of transition sequence
        self.softmax = nn.Softmax(dim=0)
        self.logsoftmax = nn.LogSoftmax(dim=0)

        if self.use_tokenizer:
            self._init_tokenizer()

    def __call__(self, sentence):
        pass

    def _init_tokenizer(self):
        if self.verbose: print("Using spaCy's Engish tokenizer.")
        from spacy.lang.en import English
        from spacy.tokenizer import Tokenizer
        self.tokenizer = Tokenizer(English().vocab)

    def _tokenize(self, sentence):
        if self.verbose: print('Tokenizing sentence...')
        return [token.text for token in self.tokenizer(sentence)]

    def _process_unks(self, sentence):
        if isinstance(sentence, str):
            sentence = sentence.split()
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
        if self.use_tokenizer:
            sentence = self._tokenize(sentence)
            if self.verbose: print(f'token: {sentence}')
        sentence = self._process_unks(sentence)
        sentence_items = []
        for word in sentence:
            if self.use_char:
                index = [self.dictionary.w2i[char] for char in word]
            else:
                index = self.dictionary.w2i[word]
            sentence_items.append(Word(word, index))
        return sentence_items

    def _process_sentence(self, sentence):
        assert sentence, f'decoder received empty sentence'
        if isinstance(sentence, str):
            return self._from_string(sentence)
        elif isinstance(sentence, list) and isinstance(sentence[0], Word):
            return sentence
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

    def _store_logprob(self, logits, index):
        logprobs = self.logsoftmax(logits)
        self._logprobs.append(logprobs[index].item())

    def _compute_logprob(self):
        logprob = sum(self._logprobs)
        self._logprobs = []
        return logprob

    def _make_action(self, index):
        """Maps index to action."""
        assert index in range(3), f'invalid action index {index}'
        if index == SHIFT.index:
            return SHIFT
        elif index == REDUCE.index:
            return REDUCE
        elif index == Action.NT_INDEX:
            return NT(Nonterminal('_', -1))

    def _valid_actions_mask(self):
        mask = torch.Tensor(
            [self.model.is_valid_action(self._make_action(i)) for i in range(3)]
        )
        return mask

    def _best_valid_action(self, logits, k=1):
        mask = self._valid_actions_mask()
        masked_logits = torch.Tensor(
            [logit if allowed else -np.inf for logit, allowed in zip(logits, mask)]
        )
        masked_logits, ids = masked_logits.sort(descending=True)
        if k == 1:
            index = ids[0]
            action = self._make_action(index)
            print(action)
            return index, action
        else:
            assert not k > logits.size(0), f'logits size {logits.shape} but k is {k}'
            print(ids)
            print(mask)
            indices = [i.item() for i in ids[:k] if mask[i]]
            print(indices)
            return indices, [self._make_action(i) for i in indices]

    def load_model(self, path):
        with open(path, 'rb') as f:
            if self.verbose: print(f'Loading model from {path}...')
            state = torch.load(f)
            epoch, fscore = state['epoch'], state['test-fscore']
            if self.verbose: print(f'Loaded model trained for {epoch} epochs with test-fscore {fscore}.')
            self.model = state['model']
            self.dictionary = state['dictionary']
            self.use_char = state['args'].use_char
            self.model.eval()  # To be sure dropout is disabled.

    def get_tree(self):
        assert len(self.model.stack._items) > 1, 'no tree built yet'
        return self.model.stack._items[1]  # Root node.


class GreedyDecoder(Decoder):
    """Greedy decoder for RNNG."""
    def __call__(self, sentence):
        with torch.no_grad():
            self.model.eval()
            sentence = self._process_sentence(sentence)
            self.model.initialize(sentence)
            while not self.model.stack.is_empty():
                # Compute action logits.
                x = self.model.get_input()
                action_logits = self.model.action_mlp(x).squeeze(0)
                index, action = self._best_valid_action(action_logits)
                self._store_logprob(action_logits, index)
                if action.is_nt:
                    nt_logits = self.model.nonterminal_mlp(x).squeeze(0)
                    nt_logits, ids = nt_logits.sort(descending=True)
                    index = ids[0]
                    nt = self.dictionary.i2n[index]
                    X = Nonterminal(nt, index)
                    action = NT(X)
                    self._store_logprob(nt_logits, index)
                self.model.parse_step(action)
            return self.get_tree(), self._compute_logprob()


class SamplingDecoder(Decoder):
    """Ancestral sampling decoder for RNNG."""
    def __call__(self, sentence, alpha=1.0):
        self.model.eval()
        sentence = self._process_sentence(sentence)
        self.model.initialize(sentence)
        while not self.model.stack.is_empty():
            # Compute action logits.
            x = self.model.get_input()
            action_logits = self.model.action_mlp(x).squeeze(0)  # tensor (num_actions)
            mask = self._valid_actions_mask()
            action_probs = self._compute_probs(action_logits, mask=mask, alpha=alpha)
            # Sample action.
            index = np.random.choice(
                range(action_probs.size(0)), p=action_probs.data.numpy()
            )
            action = self._make_action(index)
            self._store_logprob(action_logits, index)
            if action.is_nt:
                nt_logits = self.model.nonterminal_mlp(x).squeeze(0)  # tensor (num_nonterminals)
                nt_probs = self._compute_probs(nt_logits, alpha)
                # Sample nonterminal.
                index = np.random.choice(
                    range(nt_probs.size(0)), p=nt_probs.data.numpy()
                )
                X = Nonterminal(self.dictionary.i2n[index], index)
                action = NT(X)
                self._store_logprob(nt_logits, index)
            self.model.parse_step(action)
        return self.get_tree(), self._compute_logprob()


class BeamSearchDecoder(Decoder):
    """Beam search decoder for RNNG."""
    def __call__(self, sentence, k=10):
        with torch.no_grad():
            self.k = k
            sentence = self._process_sentence(sentence)
            parser = Parser(
                word_embedding=self.model.history.word_embedding,
                nt_embedding=self.model.history.nt_embedding,
                action_embedding=self.model.history.action_embedding,
                stack_encoder=self.model.stack.encoder,
                buffer_encoder=self.model.buffer.encoder,
                history_encoder=self.model.history.encoder,
                device=self.model.device
            )
            parser.eval()

            # TODO: this could be done inside parser.
            parser.stack.empty_emb = self.model.stack.empty_emb
            parser.buffer.empty_emb = self.model.buffer.empty_emb
            parser.history.empty_emb = self.model.history.empty_emb

            # self.beams = [(deepcopy(parser), 0.0) for _ in range(k)]  # (beam_state, score)
            self.beams = [(parser, 0.0)]  # (beam_state, score)
            self.finished = []
            for parser, _ in self.beams:
                parser.initialize(sentence)

            while self.beams:
                self.beam_step()
                print('finished beams:', len(self.finished))
                print('num beams:', len(self.beams))
                for parser, prob in self.beams:
                    print(parser.stack, prob.item())

            finished = [(parser.stack._items[1], logprob) for parser, logprob in self.finished]
            return sorted(finished, key=lambda x: x[1], reverse=True)

    def _best_k_valid_actions(self, parser, logits):
        k = max(self.k, logits.size(0))
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

    def beam_step(self):
        """Advance each beam one step."""
        new_beams = []
        for parser, log_prob in self.beams:
            # Compute action logits.
            x = self.get_input(parser)
            action_logits = self.model.action_mlp(x).squeeze(0)
            action_logprobs = self.logsoftmax(action_logits)
            indices, actions = self._best_k_valid_actions(parser, action_logits)
            for i, action in enumerate(actions):
                new_parser = deepcopy(parser)
                new_log_prob = log_prob + action_logprobs[i]
                if action.is_nt:
                    nt_logits = self.model.nonterminal_mlp(x).squeeze(0)
                    nt_logits, ids = nt_logits.sort(descending=True)
                    nt_logprob = self.logsoftmax(nt_logits)
                    k = self.k - len(indices) + 1  # can open this many Nonterminals.
                    for i, index in enumerate(ids[:k]):
                        new_parser = deepcopy(parser)
                        nt = self.dictionary.i2n[index]
                        X = Nonterminal(nt, index)
                        action = NT(X)
                        new_parser.parse_step(action)
                        new_beams.append((new_parser, new_log_prob + nt_logprob[i]))  # nt_logprobs has the same order as ids!
                    print()
                else:
                    new_parser.parse_step(action)
                    new_beams.append((new_parser, new_log_prob))
            del parser
        # new_beams = sorted(new_beams, key=lambda x: x[1])
        new_beams = sorted(new_beams, key=lambda x: x[1])[-self.k:]
        self.finished += [(beam, logprob) for beam, logprob in new_beams if beam.stack.is_empty()]
        self.beams = [(beam, logprob) for beam, logprob in new_beams if not beam.stack.is_empty()]



if __name__ == '__main__':
    sentence = u'The hungry banker eats .'

    greedy = GreedyDecoder()
    greedy.load_model(path='checkpoints/20180815_170655/model.pt')

    beam = BeamSearchDecoder()
    beam.load_model(path='checkpoints/20180815_170655/model.pt')

    sampler = SamplingDecoder()
    sampler.load_model(path='checkpoints/20180815_170655/model.pt')

    print('Beam-search decoder:')
    results = beam(sentence, k=5)
    for tree, logprob in results:
        print('{} {:.2f}'.format(tree.linearize(with_tag=False), logprob))
    print()

    quit()

    print('Greedy decoder:')
    tree, logprob = greedy(sentence)
    print('{} {:.2f}'.format(tree.linearize(with_tag=False), logprob))
    print()



    print('Sampling decoder:')
    for _ in range(5):
        tree, logprob = sampler(sentence)
        print('{} {:.2f}'.format(tree.linearize(with_tag=False), logprob))
    print('-'*79)
    print()
