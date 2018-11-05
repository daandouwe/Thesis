from typing import NamedTuple

import dynet as dy
from dynet import Expression
import numpy as np

from actions import SHIFT, REDUCE, NT, GEN, is_nt, is_gen, get_nt, get_word
from tree import Node, InternalNode, LeafNode


class StackElement(NamedTuple):
    id: int
    emb: Expression
    subtree: Node
    is_open_nt: bool


class Stack:

    def __init__(self, model, dictionary, word_embedding, nt_embedding, encoder, composer):
        word_embedding_dim = word_embedding.shape()[1]
        nt_embedding_dim = nt_embedding.shape()[1]
        assert (word_embedding_dim == nt_embedding_dim)

        self.dictionary = dictionary
        self.embedding_dim = word_embedding_dim
        self.word_embedding = word_embedding
        self.nt_embedding = nt_embedding
        self.encoder = encoder
        self.composer = composer
        self.empty_emb = model.add_parameters(self.embedding_dim, init='glorot')
        self._stack = []
        self._num_open_nts = 0

    def state(self):
        return f'Stack ({self._num_open_nts} open nt): {self.get_tree()}'

    def initialize(self):
        self._stack = []
        self._num_open_nts = 0
        self.encoder.initialize()
        self.encoder.push(self.empty_emb)

    def open(self, nt_id: int):
        emb = self.nt_embedding[nt_id]
        self.encoder.push(emb)
        subtree = InternalNode(self.dictionary.i2n[nt_id])
        self.attach_subtree(subtree)
        self._stack.append(StackElement(nt_id, emb, subtree, True))
        self._num_open_nts += 1

    def push(self, word_id: int):
        emb = self.word_embedding[word_id]
        self.encoder.push(emb)
        subtree = LeafNode(self.dictionary.i2w[word_id])
        self.attach_subtree(subtree)
        self._stack.append(StackElement(word_id, emb, subtree, False))

    def pop(self):
        return self._stack.pop()

    def attach_subtree(self, subtree: Node):
        """Add subtree to rightmost open nonterminal as rightmost child."""
        for node in self._stack[::-1]:
            if node.is_open_nt:
                node.subtree.add_child(subtree)
                break

    def reduce(self):
        # Gather children.
        children = []
        while not self._stack[-1].is_open_nt:
            children.append(self.pop())
        children.reverse()
        # Get head.
        head = self.pop()
        # Gather child embeddings.
        sequence_len = len(children) + 1
        # Compute new representation.
        reduced_emb = self.composer(head.emb, [child.emb for child in children])
        # Pop hidden states from StackLSTM.
        for _ in range(sequence_len):
            self.encoder.pop()
        # Reencode with reduce embedding.
        self.encoder.push(reduced_emb)
        self._stack.append(StackElement(head.id, reduced_emb, head.subtree, False))
        self._num_open_nts -= 1

    def get_tree(self):
        if self.is_empty():
            return '()'
        else:
            return self._stack[0].subtree

    def is_empty(self):
        return len(self._stack) == 0

    def is_finished(self):
        if self.is_empty():
            return False
        else:
            return not self._stack[0].is_open_nt # (S needs to be closed)

    @property
    def num_open_nts(self):
        return self._num_open_nts


class Buffer:

    def __init__(self, model, dictionary, embedding, encoder):
        self.dictionary = dictionary
        self.embedding_dim = embedding.shape()[1]
        self.embedding = embedding
        self.encoder = encoder
        self.empty_emb = model.add_parameters(self.embedding_dim, init='glorot')
        self._buffer = []

    def state(self):
        words = [self.dictionary.i2w[word_id] for word_id in self._buffer]
        return f'Buffer: {words}'

    def initialize(self, sentence: list):
        """Embed and encode the sentence."""
        self._buffer = []
        self.encoder.initialize()
        self.encoder.push(self.empty_emb)
        for word_id in reversed(sentence):
            self.push(word_id)

    def push(self, word_id):
        self._buffer.append(word_id)
        self.encoder.push(self.embedding[word_id])

    def pop(self):
        self.encoder.pop()
        return self._buffer.pop()

    def is_empty(self):
        return len(self._buffer) == 0


class Terminal:
    def __init__(self, model, dictionary, embedding, encoder):
        super(Terminal, self).__init__()
        self.dictionary = dictionary
        self.embedding_dim = embedding.shape()[1]
        self.embedding = embedding
        self.encoder = encoder
        self.empty_emb = model.add_parameters(self.embedding_dim, init='glorot')
        self._terminal = []

    def state(self):
        words = [self.dictionary.i2w[word_id] for word_id in self._terminal]
        return f'Terminal: {words}'

    def initialize(self):
        self._terminal = []
        self.encoder.initialize()
        self.encoder.push(self.empty_emb)

    def push(self, word_id):
        self._terminal.append(word_id)
        self.encoder.push(self.embedding[word_id])

    @property
    def empty(self):
        return len(self._terminal) == 0


class History:

    def __init__(self, model, dictionary, embedding, encoder):
        self.dictionary = dictionary
        self.embedding_dim = embedding.shape()[1]
        self.embedding = embedding
        self.encoder = encoder
        self.empty_emb = model.add_parameters(self.embedding_dim, init='glorot')
        self._history = []

    def state(self):
        actions = [self.dictionary.i2a[action_id] for action_id in self._history]
        return f'History: {actions}'

    def initialize(self):
        self._history = []
        self.encoder.initialize()
        self.encoder.push(self.empty_emb)

    def push(self, action_id):
        self._history.append(action_id)
        self.encoder.push(self.embedding[action_id])

    @property
    def actions(self):
        return self._history

    @property
    def is_empty(self):
        return len(self._history) == 0


class DiscParser:
    SHIFT_ID = 0
    REDUCE_ID = 1

    def __init__(self):
        pass

    def state(self):
        return '\n'.join(
            (f'Discriminative parser', self.stack.state(), self.buffer.state(), self.history.state()))

    def initialize(self, sentence: list):
        """Initialize all the components of the parser."""
        self.buffer.initialize(sentence)
        self.stack.initialize()
        self.history.initialize()

    def _can_shift(self):
        cond1 = not self.buffer.is_empty()
        cond2 = self.stack.num_open_nts >= 1
        return cond1 and cond2

    def _can_open(self):
        cond1 = not self.buffer.is_empty()
        cond2 = self.stack.num_open_nts < 100
        return cond1 and cond2

    def _can_reduce(self):
        cond1 = not self.last_action_is_nt()
        cond2 = self.stack.num_open_nts >= 2
        cond3 = self.buffer.is_empty()
        return cond1 and (cond2 or cond3)

    def _shift(self):
        assert self._can_shift(), f'cannot shift:\n{self.state()}'
        self.stack.push(self.buffer.pop())

    def _open(self, nt_index: int):
        assert self._can_open(), f'cannot open:\n{self.state()}'
        self.stack.open(nt_index)

    def _reduce(self):
        assert self._can_reduce(), f'cannot reduce:\n{self.state()}'
        self.stack.reduce()

    def parser_representation(self):
        """Return the representations of the stack, buffer and history."""
        s = self.stack.encoder.top
        b = self.buffer.encoder.top
        h = self.history.encoder.top
        return dy.concatenate([s, b, h], d=0)

    def parse_step(self, action: str, action_id: int):
        """Updates parser one step give the action."""
        if action == SHIFT:
            self._shift()
        elif action == REDUCE:
            self._reduce()
        else:
            self._open(self._get_nt_id(action_id))
        self.history.push(action_id)

    def _is_valid_action(self, action: str):
        """Check whether the action is valid under the parser's configuration."""
        if action == SHIFT:
            return self._can_shift()
        elif action == REDUCE:
            return self._can_reduce()
        else:
            return self._can_open()

    def _add_actions_mask(self):
        """Return additive mask for invalid actions."""
        return np.array(
            [-np.inf if not self._is_valid_action(self.dictionary.i2a[i]) else 0
                for i in range(self.num_actions)])

    def _mult_actions_mask(self):
        """Return additive mask for invalid actions."""
        return np.array(
            [self._is_valid_action(self.dictionary.i2a[i]) for i in range(self.num_actions)])

    def _is_nt_id(self, action_id: int):
        return action_id >= 2

    def _get_nt_id(self, action_id: int):
        assert self._is_nt_id(action_id)
        return action_id - 2

    def last_action_is_nt(self):
        if len(self.history._history) == 0:
            return False
        else:
            return self._is_nt_id(self.last_action)

    def get_tree(self):
        return self.stack.get_tree()

    @property
    def last_action(self):
        """Return the last action taken."""
        assert len(self.history.actions) > 0, 'no actions yet'
        return self.history.actions[-1]


class GenParser:
    REDUCE_ID = 0
    NT_ID = 1
    GEN_ID = 2

    def __init__(self):
        pass

    def state(self):
        return '\n'.join(
            (f'Generative parser', self.stack.state(), self.terminal.state(), self.history.state()))

    def initialize(self):
        """Initialize all the components of the parser."""
        self.terminal.initialize()
        self.stack.initialize()
        self.history.initialize()

    def _can_gen(self):
        return self.stack.num_open_nts >= 1

    def _can_open(self):
        return self.stack.num_open_nts < 100

    def _can_reduce(self):
        cond1 = not self.last_action_is_nt()
        cond2 = self.stack.num_open_nts >= 1
        return cond1 and cond2

    def _gen(self, word_id):
        assert self._can_gen(), f'cannot gen:\n{self.state()}'
        self.terminal.push(word_id)
        self.stack.push(word_id)

    def _open(self, nt_index: int):
        assert self._can_open(), f'cannot open:\n{self.state()}'
        self.stack.open(nt_index)

    def _reduce(self):
        assert self._can_reduce(), f'cannot reduce:\n{self.state()}'
        self.stack.reduce()

    def parser_representation(self):
        """Return the representations of the stack, buffer and history."""
        s = self.stack.encoder.top
        t = self.terminal.encoder.top
        h = self.history.encoder.top
        return dy.concatenate([s, t, h], d=0)

    def parse_step(self, action: str, action_id: int):
        """Updates parser one step give the action."""
        if action == REDUCE:
            self._reduce()
        elif is_nt(action):
            self._open(self._get_nt_id(action_id))
        elif is_gen(action):
            self._gen(self._get_word_id(action_id))
        self.history.push(action_id)

    def _is_valid_action(self, action: str):
        """Check whether the action is valid under the parser's configuration."""
        if action == REDUCE:
            return self._can_reduce()
        elif is_nt(action):
            return self._can_open()
        elif is_gen(action):
            return self._can_gen()
        else:
            raise ValueError("received invallid action '{action}'")

    def _add_actions_mask(self):
        """Return additive mask for invalid actions."""
        return np.array(
            [-np.inf if not self._is_valid_action(self.dictionary.i2a[i]) else 0
                for i in range(self.num_actions)])

    def _mult_actions_mask(self):
        """Return additive mask for invalid actions."""
        return np.array(
            [self._is_valid_action(action) for action in (REDUCE, NT('x'), GEN('x'))])  # dummy actions

    def _is_nt_id(self, action_id: int):
        return 0 < action_id <= self.num_nt

    def _is_gen_id(self, action_id: int):
        return action_id > self.num_nt

    def _get_nt_id(self, action_id: int):
        assert self._is_nt_id(action_id)
        return action_id - 1

    def _get_word_id(self, action_id: int):
        assert self._is_gen_id(action_id)
        return action_id - (1 + self.num_nt)

    def _get_action_id(self, action_id):
        if self._is_nt_id(action_id):
            return self.NT_ID
        elif self._is_gen_id(action_id):
            return self.GEN_ID
        else:
            return self.REDUCE_ID

    def _make_action_id_from_nt_id(self, nt_id):
        assert nt_id in range(self.num_nt)
        return nt_id + 1

    def _make_action_id_from_word_id(self, word_id):
        assert word_id in range(self.num_words)
        return word_id + self.num_nt + 1

    def last_action_is_nt(self):
        if len(self.history.actions) == 0:
            return False
        else:
            return self._is_nt_id(self.last_action)

    def get_tree(self):
        return self.stack.get_tree()

    @property
    def last_action(self):
        """Return the last action taken."""
        assert len(self.history.actions) > 0, 'no actions yet'
        return self.history.actions[-1]
