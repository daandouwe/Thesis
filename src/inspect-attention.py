import sys
from copy import deepcopy
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from PYEVALB import parser, scorer

from datatypes import Token, Item, Word, Nonterminal, Action
from actions import REDUCE, NT, GEN
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

    def forward(self, sentence, actions):
        """Forward pass only used for training."""
        sentence = self._process_sentence(sentence)
        actions = self._process_actions(actions)
        self.model.initialize(sentence)
        loss = torch.zeros(1, device=self.model.device)
        for i, action in enumerate(actions):
            # Compute loss
            x = self.model.get_input()
            action_logits = self.model.action_mlp(x)
            loss += self.model.loss_compute(action_logits, action.action_index)
            # If we open a nonterminal, predict which.
            if action.is_nt:
                nonterminal_logits = self.model.nonterminal_mlp(x)
                nt = action.get_nt()
                loss += self.model.loss_compute(nonterminal_logits, nt.index)
            self.model.parse_step(action)
            if action == REDUCE:
                # We stored these internally
                attention = self.model.stack.encoder.composition.attn
                children = self.model.stack.child_tokens
                head = self.model.stack.head_token
                print()
                print(head, '|', ' '.join(children))
                print(attention.squeeze().data.numpy())
                print()
        return loss

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


def main(path):
    sentence, actions = LONG['sentence'], LONG['actions']
    print('>', sentence)
    decoder = AttentionInspection(use_tokenizer=True)
    decoder.load_model(path)
    sentence = decoder.forward(sentence, actions)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
        main(path)
    else:
        exit('Specify path')
