import torch
import torch.nn as nn

from data import (EMPTY_INDEX, REDUCED_INDEX, EMPTY_TOKEN, REDUCED_TOKEN, PAD_TOKEN,
                    Item, Action, wrap, pad)

class TransitionBase:
    """A base class for the Stack, Buffer and History."""
    def __init__(self):
        self._items = [] # Will hold a list of Items.
        self._reset()

    def __str__(self):
        return f'{type(self).__name__}: {self.tokens}'

    def _reset(self):
        self._items = []

    @property
    def tokens(self):
        return [item.token for item in self._items]

    @property
    def indices(self):
        return [item.index for item in self._items]

    @property
    def embeddings(self):
        return [item.embedding for item in self._items]

    @property
    def encodings(self):
        return [item.encoding for item in self._items]

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

    @property
    def empty(self):
        pass


class Stack(TransitionBase):
    def __init__(self, word_embedding, nonterminal_embedding, device):
        """Initialize the Stack.

        Arguments:
            word_embedding (nn.Embedding): embedding function for words.
            nonterminal_embedding (nn.Embedding): embedding function for nonterminals.
            device: device on which computation is done (gpu or cpu).
        """
        super(Stack, self).__init__()
        self._num_open_nonterminals = 0 # Keep track of the nonterminals opened.
        self.word_embedding = word_embedding
        self.nonterminal_embedding = nonterminal_embedding
        # TODO: self.encoder = encoder
        self.device = device

    def __str__(self):
        return f'{type(self).__name__} ({self.num_open_nonterminals} open NTs): {self.tokens}'

    def initialize(self):
        """Initialize by pushing the `empty` item onto the stack."""
        self._reset()
        self._num_open_nonterminals = 0
        self.push(Item(EMPTY_TOKEN, EMPTY_INDEX))

    def open_nonterminal(self, item):
        """Open a new nonterminal in the tree."""
        self._num_open_nonterminals += 1
        self.push(item, nonterminal=True)

    def push(self, item, nonterminal=False, reduced=False):
        """Push a new item on the stack."""
        if not reduced:
            embedding_fn = self.nonterminal_embedding if nonterminal else self.word_embedding
            item.embedding = embedding_fn(wrap([item.index], self.device))
        self._items.append(item)

    def pop(self):
        """Pop tokens and vectors from the stack until first open nonterminal."""
        found_head = False
        children = []
        # We pop items from self._tokens till we find a nonterminal head.
        while not found_head:
            item = self._items.pop()
            children.append(item)
            # Break from while if we found a nonterminal
            if item.is_nonterminal:
                found_head = True
        # reverse the lists (we appended)
        children = children[::-1]
        head, children = children[0], children[1:]
        # Package embeddings as pytorch tensor
        children = [item.embedding.unsqueeze(0) for item in children]
        children = torch.cat(children, 1)  # tensor (batch, seq_len, emb_dim)
        head = head.embedding  # tensor (batch, emb_dim)
        # Update the number of open nonterminals
        self._num_open_nonterminals -= 1
        return children, head

    @property
    def empty(self):
        """Returns True if the stack is empty."""
        return self.indices == [EMPTY_INDEX, REDUCED_INDEX]

    @property
    def start(self):
        return self.indices == [EMPTY_INDEX]

    @property
    def num_open_nonterminals(self):
        """Return the number of nonterminal nodes in the tree that are not yet closed."""
        return self._num_open_nonterminals


class Buffer(TransitionBase):
    def __init__(self, embedding, encoder, device):
        """Initialize the Buffer.

        Arguments:
            embedding (nn.Embedding): embedding function for words on the buffer.
            encoder (nn.Module): encoder function to encode buffer contents.
            device: device on which computation is done (gpu or cpu).
        """
        super(Buffer, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.device = device

    def initialize(self, sentence):
        """Embed and encode the sentence."""
        self._reset()
        self._items = sentence[::-1] # On the buffer the sentence is reversed.
        embeddings = self.embedding(wrap(self.indices, self.device))
        encodings = self.encoder(embeddings.unsqueeze(0)) # (batch, seq, hidden_size)
        for i, item in enumerate(self._items):
            item.embedding = embeddings[i, :].unsqueeze(0)
            item.encoding = encodings[:, i ,:]

    def push(self, item):
        """Push item onto buffer."""
        item.embedding = self.embedding(wrap([item.index], self.device))
        item.encoding = self.encoder(item.embedding.unsqueeze(0)).squeeze(0)
        self._items.append(item)

    def pop(self):
        if self.empty:
            raise ValueError('trying to pop from an empty buffer')
        else:
            item = self._items.pop()
            if not self._items: # empty list
                # Push empty token.
                self.push(Item(EMPTY_TOKEN, EMPTY_INDEX))
            return item

    @property
    def empty(self):
        """Returns True if the buffer is empty."""
        return self.indices == [EMPTY_INDEX]


class History(TransitionBase):
    def __init__(self, embedding, device):
        """Initialize the History.

        Arguments:
            embedding (nn.Embedding): embedding function for actions.
            device: device on which computation is done (gpu or cpu).
        """
        super(History, self).__init__()
        self.embedding = embedding
        self.device = device

    def initialize(self):
        """Initialize the history by push the `empty` item."""
        self._reset()
        self.push(Action(EMPTY_TOKEN, EMPTY_INDEX))

    def push(self, item):
        """Push action index and vector embedding onto history."""
        item.embedding = self.embedding(wrap([item.index], self.device))
        self._items.append(item)

    @property
    def actions(self):
        return [item.symbol.token if item.is_nonterminal else item.token
                    for item in self._items[1:]] # First item in self._items is the empty item

class Parser(nn.Module):
    """The parse configuration."""
    def __init__(self, word_embedding, nonterminal_embedding, action_embedding,
                 buffer_encoder, actions, device):
        """Initialize the parser.

        Arguments:
            word_embedding: embedding function for words.
            nonterminal_embedding: embedding function for nonterminals.
            actions_embedding: embedding function for actions.
            buffer_encoder: encoder function to encode buffer contents.
            actions (tuple): tuple with indices of actions.
            device: device on which computation is done (gpu or cpu).
        """
        super(Parser, self).__init__()
        self.SHIFT, self.REDUCE, self.OPEN = actions
        self.stack = Stack(word_embedding, nonterminal_embedding, device)
        self.buffer = Buffer(word_embedding, buffer_encoder, device)
        self.history = History(action_embedding, device)

    def __str__(self):
        return '\n'.join(('Parser', str(self.stack), str(self.buffer), str(self.history)))

    def initialize(self, items):
        """Initialize all the components of the parser."""
        self.buffer.initialize(items)
        self.stack.initialize()
        self.history.initialize()

    def shift(self):
        """Shift an item from the buffer to the stack."""
        self.stack.push(self.buffer.pop())

    def get_embedded_input(self):
        """Return the representations of the stack buffer and history.

        Note: `buffer` is already the encoding of the buffer."""
        stack = self.stack.top_embedded     # [batch, word_emb_size]
        buffer = self.buffer.top_encoded    # [batch, word_lstm_hidden]
        history = self.history.top_embedded # [batch, action_emb_size]
        return stack, buffer, history

    def is_valid_action(self, action):
        """Check whether the action is valid under the parser's configuration."""
        if action.index == self.SHIFT:
            cond1 = not self.buffer.empty
            cond2 = self.stack.num_open_nonterminals > 0
            return cond1 and cond2
        elif action.index == self.REDUCE:
            cond1 = not self.last_action.index == self.OPEN
            cond2 = not self.stack.start
            cond3 = self.stack.num_open_nonterminals > 1
            cond4 = self.buffer.empty
            return (cond1 and cond2 and cond3) or cond4
        elif action.index == self.OPEN:
            cond1 = not self.buffer.empty
            cond2 = self.stack.num_open_nonterminals < 100
            return cond1 and cond2
        else:
            raise ValueError(f'got illegal action: {action.token}.')

    @property
    def actions(self):
        """Return the current history of actions."""
        return self.history.actions

    @property
    def last_action(self):
        """Return the last action taken."""
        return self.history.top_item

if __name__ == '__main__':
    stack = Stack(None, None, None, device=None)
    parser = Parser(None, None, None, None, None, device=None)
    print(stack)
    print(parser)
