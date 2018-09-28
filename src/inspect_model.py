import sys
from copy import deepcopy
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from PYEVALB import parser, scorer

from datatypes import Token, Item, Word, Nonterminal, Action
from actions import REDUCE, NT, GEN
from data import Corpus
from decode import Decoder
from tests.test_parser import SHORT, LONG


class AttentionInspection(Decoder):
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
                if action == REDUCE:
                    # We stored these internally
                    attention = self.model.stack.encoder.composition.attn
                    children = self.model.stack.children
                    head = self.model.stack.head
                    print()
                    print(head, '|', ' '.join(children))
                    print(attention.squeeze().data.numpy())
                    print()
            return self.get_tree(), logprob, num_actions

    def forward(self, sentence, actions, composition):
        """Forward pass only used for training."""
        self.model.eval()
        self.model.initialize(sentence)
        for i, action in enumerate(actions):
            # Compute loss
            x = self.model.get_input()
            action_logits = self.model.action_mlp(x)
            # If we open a nonterminal, predict which.
            if action.is_nt:
                nonterminal_logits = self.model.nonterminal_mlp(x)
                nt = action.get_nt()
            self.model.parse_step(action)
            if action == REDUCE:
                items = self.model.reduced_items()
                head, children = items['head'], items['children']
                if composition == 'attention':
                    attention = items['attention'].squeeze(0).data.numpy()
                    gate = items['gate'].squeeze(0).data.numpy()
                    attentive = [f'{child.token} ({attn:.2f})'
                        for child, attn in zip(children, attention)]
                    print('  ', head.token, '|', ' '.join(attentive), f'[{gate.mean():.2f}]')
                elif composition == 'latent-factors':
                    sample = items['sample'].squeeze(0)
                    alpha = items['alpha'].squeeze(0)
                    factors = sample.squeeze(0).data.numpy().astype(int)
                    # print('  ', head.token, '|', ' '.join(children), factors)
                    # probs = nn.functional.sigmoid(alpha).data.numpy()
                    probs = nn.functional.softmax(alpha).data.numpy()
                    print('  ', head.token, probs)

    def _process_actions(self, actions):
        action_items = []
        token_idx = 0
        for a in actions:
            if a == 'SHIFT':
                action = Action('SHIFT', Action.SHIFT_INDEX)
            elif a == 'REDUCE':
                action = Action('REDUCE', Action.REDUCE_INDEX)
            elif a.startswith('NT'):
                nt = a[3:-1]
                action = NT(Nonterminal(nt, self.dictionary.n2i[nt]))
            action_items.append(action)
        return action_items


def main(args):
    decoder = AttentionInspection(use_tokenizer=True)
    decoder.load_model(args.checkpoint)
    if True:
        print(f'Loading data from `{args.data}`...')
        corpus = Corpus(
            data_path=args.data,
            model=args.model,
            textline=args.textline,
            name=args.name,
            use_chars=args.use_chars,
            max_lines=args.max_lines
        )
        train_batches = corpus.train.batches(length_ordered=False, shuffle=True)
        dev_batches = corpus.dev.batches(length_ordered=False, shuffle=False)
        test_batches = corpus.test.batches(length_ordered=False, shuffle=False)
        for sentence, actions in train_batches[:args.max_lines]:
            print('>', ' '.join([str(token) for token in sentence]))
            decoder.forward(sentence, actions, composition=args.composition)
            print()
    else:
        sentence, actions = LONG['sentence'], LONG['actions']
        sentence = decoder._process_sentence(sentence)
        actions = decoder._process_actions(actions)
        print('>', sentence)
        sentence = decoder.forward(sentence, actions, composition=args.composition)
