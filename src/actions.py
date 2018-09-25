"""
With inspiration from kmkurn/pytorch-rnng:
    https://github.com/kmkurn/pytorch-rnng/blob/master/src/rnng/actions.py
"""
from datatypes import Action, Word, Nonterminal


SHIFT = Action('SHIFT', Action.SHIFT_INDEX)


REDUCE = Action('REDUCE', Action.REDUCE_INDEX)


def GEN(word):
    assert isinstance(word, Word), f'invalid word {word:!r}'
    token = f'GEN({word.token})'
    return Action(token, word.index, word.embedding, word.encoding)


def NT(nt):
    assert isinstance(nt, Nonterminal), f'invalid nonterminal {nt:!r}'
    token = f'NT({nt.token})'
    return Action(token, nt.index, nt.embedding, nt.encoding)
