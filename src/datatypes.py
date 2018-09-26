"""
With inspiration from kmkurn/pytorch-rnng:
    https://github.com/kmkurn/pytorch-rnng/blob/master/src/rnng/actions.py
"""
from typing import NamedTuple


class Token(NamedTuple):
    original: str
    processed: str

    def __str__(self):
        """Used when print the token in a predicted tree."""
        return self.original

    def __repr__(self):
        """Used for other purposes."""
        return repr(self)


class Item:
    def __init__(self, token, index, embedding=None, encoding=None):
        self.token = token
        self.index = index
        self.embedding = embedding
        self.encoding = encoding

    def __str__(self):
        return str(self.token)

    def __repr__(self):
        return f'{type(self).__name__}(token={repr(self.token)}, index={self.index})'

    def __eq__(self, other):
        if not type(self) == type(other):
            return False
        else:
            return self.token == other.token


class Nonterminal(Item):
    pass


class Word(Item):
    pass


class Action(Item):
    """Action can be used by both discriminative and generative parser."""
    SHIFT_INDEX = 0
    GEN_INDEX = 0  # SHIFT and GEN are mutually exclusive
    NT_INDEX = 1
    REDUCE_INDEX = 2

    def get_word(self):
        assert self.is_gen, f'cannot get word from {self:!r}'
        word = self.token[4:-1]
        return Word(word, self.index, self.embedding, self.encoding)

    def get_nt(self):
        assert self.is_nt, f'cannot get nt from {self:!r}'
        nt = self.token[3:-1]
        return Nonterminal(nt, self.index, self.embedding, self.encoding)

    @property
    def is_gen(self):
        return self.token.startswith('GEN(') and self.token.endswith(')')

    @property
    def is_nt(self):
        return self.token.startswith('NT(') and self.token.endswith(')')

    @property
    def action_index(self):
        if self.is_gen:
            return self.GEN_INDEX
        if self.is_nt:
            return self.NT_INDEX
        else:  # self is a SHIFT or REDUCE action
            return self.index


if __name__ == '__main__':
    token = Token('initially', 'UNK')
    word = Word(token, 0)

    print(token)
    print(repr(token))
    print()
    print(word)
    print(repr(word))

    word2 = Word('The', 0)
    nt = Nonterminal('NP', 4)
    print(word == word2)
    print(isinstance(word, Item))
    print(word == nt)
