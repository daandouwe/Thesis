from copy import deepcopy

import torch
import torch.nn as nn

from data import wrap
from datatypes import Item, Word, Nonterminal, Action
from actions import SHIFT, REDUCE, GEN, NT
from tree import InternalNode, LeafNode


class TransitionBase(nn.Module):
    EMPTY_TOKEN = '-EMPTY-'  # used as dummy to encode an empty buffer or history
    EMPTY_INDEX = -1

    """A base class for the Stack, Buffer, Terminals, and History."""
    def __init__(self):
        super(TransitionBase, self).__init__()
        self._items = []

    def __str__(self):
        return f'{type(self).__name__}: {self.tokens}'

    def pop(self):
        assert len(self._items) > 0
        return self._items.pop()

    @property
    def items(self):
        return self._items

    @property
    def tokens(self):
        return [str(item.token) for item in self.items]

    @property
    def indices(self):
        return [item.index for item in self.items]

    @property
    def embeddings(self):
        return [item.embedding for item in self.items]

    @property
    def encodings(self):
        return [item.encoding for item in self.items]

    @property
    def top(self):
        return self._items[-1]

    @property
    def top_item(self):
        return self._items[-1]

    @property
    def top_token(self):
        return self.top_item.token

    @property
    def top_index(self):
        return self.top_item.index

    @property
    def top_embedded(self):
        return self.top_item.embedding

    @property
    def top_encoded(self):
        return self.top_item.encoding


class Stack(TransitionBase):
    def __init__(self, word_embedding, nt_embedding, encoder, device):
        """Initialize the Stack.

        Arguments:
            word_embedding (nn.Embedding): embedding function for words.
            nt_embedding (nn.Embedding): embedding function for nonterminals.
            encoder (nn.Module): recurrent encoder.
            device: device on which computation is done (gpu or cpu).
        """
        super(Stack, self).__init__()
        assert (word_embedding.embedding_dim == nt_embedding.embedding_dim)
        self.embedding_dim = word_embedding.embedding_dim
        self.word_embedding = word_embedding
        self.nt_embedding = nt_embedding
        self.encoder = encoder
        self.device = device
        self.num_open_nonterminals = 0
        self.training = True
        self.empty_emb = nn.Parameter(torch.zeros(1, self.embedding_dim, device=self.device))

    def __str__(self):
        return f'Stack ({self.num_open_nonterminals} open NTs): {self.tokens}'

    def _reset(self):
        self.encoder.initialize()
        empty = Item(
            self.EMPTY_TOKEN, self.EMPTY_INDEX, self.empty_emb, self.encoder(self.empty_emb))
        self._items = [InternalNode(empty)]

    def initialize(self):
        self._reset()
        self.num_open_nonterminals = 0

    def open(self, nt):
        assert isinstance(nt, Nonterminal), nt
        nt.embedding = self.nt_embedding(wrap([nt.index], self.device))
        self.push(nt)
        self.num_open_nonterminals += 1

    def push(self, item):
        assert isinstance(item, Item), item
        token = item.token
        index = item.index
        if isinstance(item, Word):
            embedding = self.word_embedding(wrap([index], self.device))
            encoding = self.encoder(embedding)
            item = Word(token, index, embedding, encoding)
            node = LeafNode(item)
        elif isinstance(item, Nonterminal):
            embedding = self.nt_embedding(wrap([index], self.device))
            encoding = self.encoder(embedding)
            item = Nonterminal(token, index, embedding, encoding)
            node = InternalNode(item)
        else:
            raise ValueError(f'invalid {item} pushed onto stack')
        # Add child node to rightmost open nonterminal.
        for head in self._items[::-1]:
            if head.is_open_nt:
                head.add_child(node)
                break
        self._items.append(node)

    def reduce(self):
        # Gather children.
        children = []
        while not self.top.is_open_nt:
            children.append(self.pop())
        children.reverse()
        sequence_len = len(children) + 1
        head = self.pop()
        # Compute new representation.
        child_embeddings = [child.item.embedding.unsqueeze(0) for child in children]
        child_embeddings = torch.cat(child_embeddings, 1)  # tensor (batch, seq_len, emb_dim)
        reduced = self.encoder.composition(head.item.embedding, child_embeddings)
        if not self.training:
            # Store these for inspection during prediction.
            self._reduced_head_item = head.item
            self._reduced_child_items = [child.item for child in children]
            self._reduced_embedding = reduced.data
        # Set new representation.
        head.item.embedding = reduced
        self.reset_hidden(sequence_len)
        head.item.encoding = self.encoder(reduced)
        head.close()
        self.num_open_nonterminals -= 1
        self._items.append(head)

    def reset_hidden(self, sequence_len):
        self.encoder._reset_hidden(sequence_len)

    def get_tree(self, with_tag=True):
        assert not self.training, f'set model.eval() to build tree'
        # Build tree from first node, and hence skip the dummy empty node.
        return self._items[1].linearize(with_tag)

    def is_empty(self):
        if len(self._items) == 2:
            return not self.top.is_open_nt  # e.g. [-EMPTY-, S] (S needs to be closed)
        else:
            return False # e.g. [-EMPTY-] (start) or [-EMPTY-, S, NP, ...] (not finished)

    @property
    def empty(self):
        return self.is_empty()  # TODO wtf

    @property
    def items(self):
        return [node.item for node in self._items]

    @property
    def top_item(self):
        return self.items[-1]

    @property
    def top_embedded(self):
        items = self.items
        top = self.items[-1].item.embedding
        return top

    @property
    def top_encoded(self):
        items = self.items
        top = self.items[-1].item.encoding
        return top


class Buffer(TransitionBase):
    def __init__(self, embedding, encoder, device):
        """Initialize the Buffer.

        Arguments:
            embedding (nn.Embedding): embedding function for words on the buffer.
            encoder (nn.Module): encoder function to encode buffer contents.
            device: device on which computation is done (gpu or cpu).
        """
        super(Buffer, self).__init__()
        self.embedding_dim = embedding.embedding_dim
        self.embedding = embedding
        self.encoder = encoder
        self.device = device
        self.empty_emb = nn.Parameter(torch.zeros(1, self.embedding_dim, device=self.device))

    def _reset(self):
        empty = Action(
            self.EMPTY_TOKEN, self.EMPTY_INDEX, embedding=self.empty_emb)
        self._items = [empty]

    def initialize(self, sentence):
        """Embed and encode the sentence."""
        self._reset()
        self._items += sentence[::-1]
        empty_embedding = self.items[0].embedding  # (1, emb_dim)
        # Embed items without the first element, which is EMPTY and already embedded.
        embeddings = self.embedding(wrap(self.indices[1:], self.device))  # (seq_len, emb_dim)
        embeddings = torch.cat((empty_embedding, embeddings), dim=0).unsqueeze(0)  # (1, seq_len+1, emb_dim)
        # Encode everything together.
        encodings = self.encoder(embeddings)  # (1, seq_len+1, hidden_size)
        for i, item in enumerate(self._items):
            item.embedding = embeddings[:, i, :]  # (1, emb_dim)
            item.encoding = encodings[:, i, :]  # (1, hidden_size)
        del empty_embedding, embeddings, encodings

    @property
    def empty(self):
        return len(self._items) == 1


class Terminals(TransitionBase):
    def __init__(self, word_embedding, encoder, device):
        super(Terminals, self).__init__()
        self.word_embedding = word_embedding
        self.embedding_dim = word_embedding.embedding_dim
        self.encoder = encoder
        self.device = device
        self.empty_emb = nn.Parameter(torch.zeros(1, self.embedding_dim, device=device))

    def _reset(self):
        self.encoder.initialize()
        empty = Word(
            self.EMPTY_TOKEN, self.EMPTY_INDEX, self.empty_emb, self.encoder(self.empty_emb))
        self._items = [empty]

    def initialize(self):
        self._reset()

    def push(self, word):
        assert isinstance(word, Word), word
        word.embedding = self.word_embedding(wrap([word.index], self.device))
        word.encoding = self.encoder(word.embedding)
        self._items.append(word)

    @property
    def empty(self):
        return len(self._items) == 1


class History(TransitionBase):
    def __init__(self, word_embedding, nt_embedding, action_embedding, encoder, device):
        """Initialize the History.

        Arguments:
            embedding (nn.Embedding): embedding function for actions.
            device: device on which computation is done (gpu or cpu).
        """
        super(History, self).__init__()
        assert word_embedding.embedding_dim == action_embedding.embedding_dim == nt_embedding.embedding_dim
        self.embedding_dim = word_embedding.embedding_dim
        self.word_embedding = word_embedding
        self.nt_embedding = nt_embedding
        self.action_embedding = action_embedding
        self.encoder = encoder
        self.device = device
        self.empty_emb = nn.Parameter(torch.zeros(1, self.embedding_dim, device=device))

    def _reset(self):
        self.encoder.initialize()
        empty = Action(
            self.EMPTY_TOKEN, self.EMPTY_INDEX, self.empty_emb, self.encoder(self.empty_emb))
        self._items = [empty]

    def initialize(self):
        self._reset()

    def push(self, action):
        assert isinstance(action, Action), f'invalid action {action}'
        # Embed the action.
        if action.is_nt:
            nt = action.get_nt()
            action.embedding = self.nt_embedding(wrap([nt.index], self.device))
        elif action.is_gen:
            word = action.get_word()
            action.embedding = self.word_embedding(wrap([word.index], self.device))
        else:  # Shift or Reduce
            action.embedding = self.action_embedding(wrap([action.index], self.device))
        # Encode the action.
        action.encoding = self.encoder(action.embedding)
        self._items.append(action)

    @property
    def actions(self):
        return [token for token in self.items[1:]]  # First item in self._items is the empty item

    @property
    def empty(self):
        return len(self._items) == 1


class DiscParser(nn.Module):
    """The parse configuration."""
    def __init__(self, word_embedding, nt_embedding, action_embedding,
                 stack_encoder, buffer_encoder, history_encoder, device=None):
        """Initialize the parser.

        Arguments:
            word_embedding: embedding function for words.
            nt_embedding: embedding function for nonterminals.
            actions_embedding: embedding function for actions.
            buffer_encoder: encoder function to encode buffer contents.
            actions (tuple): tuple with indices of actions.
            device: device on which computation is done (gpu or cpu).
        """
        super(DiscParser, self).__init__()
        self.stack = Stack(word_embedding, nt_embedding, stack_encoder, device)
        self.buffer = Buffer(word_embedding, buffer_encoder, device)
        self.history = History(word_embedding, nt_embedding, action_embedding, history_encoder, device)

    def __str__(self):
        return '\n'.join(('Parser', str(self.stack), str(self.buffer), str(self.history)))

    def initialize(self, sentence):
        """Initialize all the components of the parser."""
        self.buffer.initialize(sentence)
        self.stack.initialize()
        self.history.initialize()
        self.stack.training = self.training

    def _can_shift(self):
        cond1 = not self.buffer.empty
        cond2 = self.stack.num_open_nonterminals >= 1
        return cond1 and cond2

    def _can_open(self):
        cond1 = not self.buffer.empty
        cond2 = self.stack.num_open_nonterminals < 100
        return cond1 and cond2

    def _can_reduce(self):
        cond1 = not self.last_action.is_nt
        cond2 = self.stack.num_open_nonterminals >= 2
        cond3 = self.buffer.empty
        return (cond1 and cond2) or cond3

    def _shift(self):
        assert self._can_shift(), f'cannot shift: {self}'
        self.stack.push(self.buffer.pop())

    def _open(self, nt):
        assert isinstance(nt, Nonterminal), nt
        assert self._can_open(), f'cannot open: {self}'
        self.stack.open(nt)

    def _reduce(self):
        assert self._can_reduce(), f'cannot reduce: {self}'
        self.stack.reduce()

    def get_encoded_input(self):
        """Return the representations of the stack, buffer and history."""
        # TODO AttributeError: 'Stack' object has no attribute 'top_encoded'.
        # stack = self.stack.top_encoded      # (batch, word_lstm_hidden)
        stack = self.stack.top_item.encoding  # (batch, word_lstm_hidden)
        buffer = self.buffer.top_encoded      # (batch, word_lstm_hidden)
        history = self.history.top_encoded    # (batch, action_lstm_hidden)
        return stack, buffer, history

    def parse_step(self, action):
        """Updates parser one step give the action."""
        assert isinstance(action, Action), action
        if action == SHIFT:
            self._shift()
        elif action == REDUCE:
            self._reduce()
        elif action.is_nt:
            self._open(action.get_nt())
        else:
            raise ValueError(f'got illegal action: {action}')
        self.history.push(action)

    def is_valid_action(self, action):
        """Check whether the action is valid under the parser's configuration."""
        assert isinstance(action, Action), action
        if action == SHIFT:
            return self._can_shift()
        elif action == REDUCE:
            return self._can_reduce()
        elif action.is_nt:
            return self._can_open()
        else:
            raise ValueError(f'got illegal action: {action}')

    @property
    def actions(self):
        """Return the current history of actions."""
        return self.history.actions

    @property
    def last_action(self):
        """Return the last action taken."""
        return self.history.top


class GenParser(nn.Module):
    """The parse configuration."""
    def __init__(self, word_embedding, nt_embedding, action_embedding,
                 stack_encoder, terminal_encoder, history_encoder, device=None):
        """Initialize the parser.

        Arguments:
            word_embedding: embedding function for words.
            nt_embedding: embedding function for nonterminals.
            actions_embedding: embedding function for actions.
            terminal_encoder: encoder function to encode buffer contents.
            actions (tuple): tuple with indices of actions.
            device: device on which computation is done (gpu or cpu).
        """
        super(GenParser, self).__init__()
        self.stack = Stack(word_embedding, nt_embedding, stack_encoder, device)
        self.terminals = Terminals(word_embedding, terminal_encoder, device)
        self.history = History(word_embedding, nt_embedding, action_embedding, history_encoder, device)

    def __str__(self):
        return '\n'.join(('Parser', str(self.stack), str(self.terminals), str(self.history)))

    def initialize(self):
        """Initialize all the components of the parser."""
        self.stack.initialize()
        self.terminals.initialize()
        self.history.initialize()
        self.stack.training = self.training

    def _can_gen(self):
        return self.stack.num_open_nonterminals >= 1

    def _can_open(self):
        return self.stack.num_open_nonterminals < 100

    def _can_reduce(self):
        cond1 = not self.last_action.is_nt
        cond2 = self.stack.num_open_nonterminals >= 1
        return cond1 and cond2

    def _gen(self, word):
        assert isinstance(word, Word), word
        assert self._can_gen(), f'cannot gen: {self}'
        self.terminals.push(word)
        self.stack.push(word)

    def _open(self, nt):
        assert isinstance(nt, Nonterminal), nt
        assert self._can_open(), f'cannot open: {self}'
        self.stack.open(nt)

    def _reduce(self):
        assert self._can_reduce(), f'cannot reduce: {self}'
        self.stack.reduce()

    def get_encoded_input(self):
        """Return the representations of the stack, buffer and history."""
        # TODO AttributeError: 'Stack' object has no attribute 'top_encoded'.
        # stack = self.stack.top_encoded            # (batch, word_lstm_hidden)
        stack = self.stack.top_item.encoding        # (batch, word_lstm_hidden)
        terminals = self.terminals.top_encoded      # (batch, word_lstm_hidden)
        history = self.history.top_encoded          # (batch, action_lstm_hidden)
        return stack, terminals, history

    def parse_step(self, action):
        """Updates parser one step give the action."""
        assert isinstance(action, Action), action
        if action == REDUCE:
            self._reduce()
        elif action.is_gen:
            self._gen(action.get_word())
        elif action.is_nt:
            self._open(action.get_nt())
        else:
            raise ValueError(f'got illegal action: {action}')
        self.history.push(action)

    def is_valid_action(self, action):
        """Check whether the action is valid under the parser's configuration."""
        assert isinstance(action, Action), action
        if action == REDUCE:
            return self._can_reduce()
        elif action.is_gen:
            return self._can_gen()
        elif action.is_nt:
            return self._can_open()
        else:
            raise ValueError(f'got illegal action: {action}')

    @property
    def actions(self):
        """Return the current history of actions."""
        return self.history.actions

    @property
    def last_action(self):
        """Return the last action taken."""
        return self.history.top
