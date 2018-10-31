#!/usr/bin/env python
from typing import NamedTuple

import torch
import torch.nn as nn

import matchbox

from model import DiscRNNG


SENT = dict(
    tagged_tree='(S (NP (NNP Avco) (NNP Corp.)) (VP (VBD received) (NP (NP (DT an) (ADJP (QP ($ $) (CD 11.8) (CD million))) (NNP Army) (NN contract)) (PP (IN for) (NP (NN helicopter) (NNS engines))))) (. .))',
    tree='(S (NP Avco Corp.) (VP received (NP (NP an (ADJP (QP $ 11.8 million)) Army contract) (PP for (NP helicopter engines)))) .)',
    sentence='Avco Corp. received an $ 11.8 million Army contract for helicopter engines .'.split(),
    actions=[
        'NT(S)',
        'NT(NP)',
        'SHIFT',
        'SHIFT',
        'REDUCE',
        'NT(VP)',
        'SHIFT',
        'NT(NP)',
        'NT(NP)',
        'SHIFT',
        'NT(ADJP)',
        'NT(QP)',
        'SHIFT',
        'SHIFT',
        'SHIFT',
        'REDUCE',
        'REDUCE',
        'SHIFT',
        'SHIFT',
        'REDUCE',
        'NT(PP)',
        'SHIFT',
        'NT(NP)',
        'SHIFT',
        'SHIFT',
        'REDUCE',
        'REDUCE',
        'REDUCE',
        'REDUCE',
        'SHIFT',
        'REDUCE'
    ]
)


def NT(nt):
    return f'NT({nt})'

def is_nt(action):
    return action.startswith('NT(') and action.endswith(')')


def get_nt(action):
    assert is_nt(action)
    return action[3:-1]


class Dictionary(NamedTuple):
    i2w: dict
    i2a: dict
    i2n: dict
    w2i: dict
    a2i: dict
    n2i: dict


def main():
    word_vocab = list(sorted(set(SENT['sentence'])))
    nt_vocab = list(sorted(set([get_nt(action) for action in SENT['actions'] if is_nt(action)])))
    action_vocab = ['SHIFT', 'REDUCE'] + [NT(nt) for nt in nt_vocab]

    i2w = word_vocab
    i2a = action_vocab
    i2n = nt_vocab
    w2i = dict(zip(word_vocab, range(len(word_vocab))))
    a2i = dict(zip(action_vocab, range(len(action_vocab))))
    n2i = dict(zip(nt_vocab, range(len(nt_vocab))))

    dictionary = Dictionary(
        i2w,
        i2a,
        i2n,
        w2i,
        a2i,
        n2i
    )
    print(w2i)
    print(a2i)
    print(n2i)

    words = torch.tensor([dictionary.w2i[w] for w in SENT['sentence']]).unsqueeze(0)
    actions = torch.tensor([dictionary.a2i[a] for a in SENT['actions']]).unsqueeze(0)

    print(words)
    print(actions)

    model = DiscRNNG(
        dictionary=dictionary,
        num_words=len(w2i),
        num_nt=len(n2i),
        word_emb_dim=10,
        nt_emb_dim=10,
        action_emb_dim=10,
        stack_emb_dim=10,
        buffer_emb_dim=10,
        history_emb_dim=10,
        stack_hidden_size=10,
        buffer_hidden_size=10,
        history_hidden_size=10,
        stack_num_layers=2,
        buffer_num_layers=2,
        history_num_layers=2,
        mlp_hidden=10,
        dropout=0.3,
        device=None
    )

    llh = model(words, actions)
    loss = -llh



if __name__ == '__main__':
    main()
