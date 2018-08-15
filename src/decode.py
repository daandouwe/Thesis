import json

import numpy as np
import torch
import torch.nn as nn

from datatypes import Item, Word, Nonterminal, Action
from actions import SHIFT, REDUCE, NT, GEN
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
        self.model.eval()
        sentence = self._process_sentence(sentence)
        self.model.initialize(sentence)
        while not self.model.stack.is_empty():
            # Compute action logits.
            x = self.model.get_input()
            action_logits = self.model.action_mlp(x)

            # TODO something like:
            # mask = (1, -inf, 1) = self.model.get_illegal_actions()
            # (1.3, 0.2, -inf) = (action_logits * mask).sort(descending=True)
            # action_index = ids[0]

            # Get highest scoring valid predictions.
            action_logits, ids = action_logits.sort(descending=True)
            action_logits, ids = action_logits.data.squeeze(0), ids.data.squeeze(0)
            i = 0
            index = ids[i]
            action = self._make_action(index)
            while not self.model.is_valid_action(action):
                i += 1
                index = ids[i]
                action = self._make_action(index)
            self._store_logprob(action_logits, index)
            if action.is_nt:
                nt_logits = self.model.nonterminal_mlp(x)
                nt_logits, ids = nt_logits.sort(descending=True)
                nt_logits, ids = nt_logits.data.squeeze(0), ids.data.squeeze(0)
                X = Nonterminal(self.dictionary.i2n[ids[0]], ids[0])
                action = NT(X)
                self._store_logprob(nt_logits, ids[0])
            self.model.parse_step(action)
        return self.get_tree(), self._compute_logprob()


class SamplingDecoder(Decoder):
    """Ancestral sampling decoder for RNNG."""

    def _compute_probs(self, logits, alpha):
        # Compute probs
        probs = self.softmax(logits).data.numpy()
        # Apply temperature scaling.
        probs = probs**alpha
        # Renormalize.
        probs /= probs.sum()
        return probs

    def __call__(self, sentence, alpha=1.0):
        self.model.eval()
        sentence = self._process_sentence(sentence)
        self.model.initialize(sentence)
        while not self.model.stack.is_empty():
            # Compute action logits.
            x = self.model.get_input()
            action_logits = self.model.action_mlp(x).squeeze(0)  # tensor (num_actions)
            action_probs = self._compute_probs(action_logits, alpha)
            # Sample action.
            index = np.random.choice(range(action_probs.shape[0]), p=action_probs)
            action = self._make_action(index)
            while not self.model.is_valid_action(action):
                # Sample new action.
                index = np.random.choice(range(action_probs.shape[0]), p=action_probs)
                action = self._make_action(index)
            self._store_logprob(action_logits, index)
            if action.is_nt:
                nt_logits = self.model.nonterminal_mlp(x).squeeze(0)  # tensor (num_nonterminals)
                nt_probs = self._compute_probs(nt_logits, alpha)
                # Sample nonterminal.
                index = np.random.choice(range(nt_probs.shape[0]), p=nt_probs)
                X = Nonterminal(self.dictionary.i2n[index], index)
                action = NT(X)
                self._store_logprob(nt_logits, index)
            self.model.parse_step(action)
        return self.get_tree(), self._compute_logprob()


class BeamSearchDecoder(Decoder):
    """Beam search decoder for RNNG."""
    def __call__(self, sentence):
        return 'not', 'yet'


if __name__ == '__main__':
    sentence = u'This is a short test sentence , but it should suffice .'

    greedy = GreedyDecoder()
    greedy.load_model(path='checkpoints/20180815_170655/model.pt')

    beam = BeamSearchDecoder()
    beam.load_model(path='checkpoints/20180815_170655/model.pt')

    sampler = SamplingDecoder()
    sampler.load_model(path='checkpoints/20180815_170655/model.pt')

    print('Beam-search decoder:')
    tree, logprob = beam(sentence)
    print('{} {:.2f}'.format(tree.linearize(with_tag=False), logprob))
    print()

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
