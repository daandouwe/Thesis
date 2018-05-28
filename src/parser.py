import torch

from data import EMPTY_INDEX, REDUCED_INDEX, EMPTY_TOKEN, REDUCED_TOKEN, PAD_TOKEN
from data import wrap

class Stack:
    """The stack"""
    def __init__(self, dictionary, embedding):
        """Initialize the Stack.

        Args:
            dictionary: an instance of data.Dictionary
        """
        self._tokens = [] # list of indices
        self._embeddings = [] # list of embeddings (pytorch vectors)
        self._num_open_nonterminals = 0

        self.dict = dictionary
        self.embedding = embedding

    def __str__(self):
        stack = [self.dict.i2s[i] for i in self._tokens]
        return 'Stack ({} open NTs): {}'.format(self.num_open_nonterminals, stack)

    def _reset(self):
        """Resets the buffer to empty state."""
        self._tokens = []
        self._embeddings = []

    def initialize(self):
        self._reset()
        self.push(EMPTY_INDEX)

    def push(self, token, vec=None, new_nonterminal=False):
        if new_nonterminal: # if we push a nonterminal onto the stack
            self._num_open_nonterminals += 1
        if vec is None: # if we did not provide a vector embedding for the token
            vec = self.embedding(wrap([token]))
        self._tokens.append(token)
        self._embeddings.append(vec)

    def pop(self):
        """Pop tokens and vectors from the stack until first open nonterminal."""
        found_nonterminal = False
        tokens, embeddings = [], []
        # We pop items from self._tokens till we find a nonterminal.
        while not found_nonterminal:
            i = self._tokens.pop()
            v = self._embeddings.pop()
            tokens.append(i)
            embeddings.append(v)
            token = self.dict.i2s[i]
            # Break from while if we found a nonterminal
            if token.startswith('NT'):
                found_nonterminal = True
        # reverse the lists
        tokens = tokens[::-1]
        embeddings = embeddings[::-1]
        # add nonterminal also to the end of both lists
        tokens.append(tokens[0])
        embeddings.append(embeddings[0])
        # Package embeddings as pytorch tensor
        embs = [emb.unsqueeze(0) for emb in embeddings]
        embeddings = torch.cat(embs, 1) # [batch, seq_len, emb_dim]
        # Update the number of open nonterminals
        self._num_open_nonterminals -= 1
        return tokens, embeddings

    @property
    def top_embedded(self):
        """Returns the embedding of the symbol on the top of the stack."""
        return self._embeddings[-1]

    @property
    def empty(self):
        return self._tokens == [EMPTY_INDEX, REDUCED_INDEX]

    @property
    def num_open_nonterminals(self):
        return self._num_open_nonterminals

class Buffer:
    """The buffer."""
    def __init__(self, dictionary, embedding):
        self._tokens = []
        self._embeddings = []

        self.dict = dictionary
        self.embedding = embedding

    def __str__(self):
        buffer = [self.dict.i2w[i] for i in self._tokens]
        return 'Buffer : {}'.format(buffer)

    def _reset(self):
        """Resets the buffer to empty state."""
        self._tokens = []
        self._embeddings = []

    def initialize(self, sentence):
        """Initialize buffer by loading in the sentence in reverse order."""
        self._reset()
        for token in sentence[::-1]:
            self.push(token)

    def pop(self):
        if self.empty:
            raise ValueError('trying to pop from an empty buffer')
        else:
            token = self._tokens.pop()
            vec = self._embeddings.pop()
            # If this pop makes the buffer empty, push
            # the empty token to signal that it is empty.
            if not self._tokens:
                self.push(EMPTY_INDEX)
            return token, vec

    def push(self, token):
        """Push action index and vector embedding onto buffer."""
        self._tokens.append(token)
        vec = self.embedding(wrap([token]))
        self._embeddings.append(vec)

    @property
    def embedded(self):
        """Concatenate all the embeddings and return as pytorch tensor"""
        embs = [emb.unsqueeze(0) for emb in self._embeddings]
        return torch.cat(embs, 1) # [batch, seq_len, emb_dim]

    @property
    def empty(self):
        return self._tokens == [EMPTY_INDEX]


class History:
    def __init__(self, dictionary, embedding):
        self._actions = []
        self._embeddings = []

        self.dict = dictionary
        self.embedding = embedding

    def __str__(self):
        history = [self.dict.i2a[i] for i in self._actions]
        return 'History : {}'.format(history)

    def _reset(self):
        """Resets the buffer to empty state."""
        self._actions = []
        self._embeddings = []

    def initialize(self):
        self._reset()
        self.push(EMPTY_INDEX)

    def push(self, action):
        """Push action index and vector embedding onto history."""
        self._actions.append(action)
        self._embeddings.append(self.embedding(wrap([action])))

    @property
    def embedded(self):
        """Concatenate all the embeddings and return as pytorch tensor"""
        embs = [emb.unsqueeze(0) for emb in self._embeddings]
        return torch.cat(embs, 1) # [batch, seq_len, emb_dim]

    @property
    def last_action(self):
        i = self._actions[-1]
        return self.dict.i2a[i]

class Parser:
    """The parse configuration."""
    def __init__(self, dictionary, embedding, history_embedding):
        self.stack = Stack(dictionary, embedding)
        self.buffer = Buffer(dictionary, embedding)
        self.history = History(dictionary, history_embedding)
        self.dict = dictionary

    def __str__(self):
        return 'PARSER STATE\n{}\n{}\n{}'.format(self.stack, self.buffer, self.history)

    def initialize(self, sentence):
        self.stack.initialize()
        self.buffer.initialize(sentence)
        self.history.initialize()

    def shift(self):
        idx, _ = self.buffer.pop()
        # Translate between dictionaries.
        token = self.dict.i2w[idx] # buffer dictionary
        idx = self.dict.s2i[token] # stack dictionary
        self.stack.push(idx)

    def get_embedded_input(self):
        stack = self.stack.top_embedded # input on top [batch, emb_size]
        buffer = self.buffer.embedded # entire buffer [batch, seq_len, emb_size]
        history = self.history.embedded # entire history [batch, seq_len, emb_size]
        return stack, buffer, history

    def is_valid_action(self, action):
        """Check whether the action is valid under the parser's configuration."""
        if action == 'SHIFT':
            cond1 = not self.buffer.empty
            cond2 = self.stack.num_open_nonterminals > 0
            return cond1 and cond2
        elif action =='REDUCE':
            cond1 = not self.history.last_action.startswith('NT')
            cond2 = self.stack.num_open_nonterminals > 1
            cond3 = self.buffer.empty
            return cond1 and (cond2 or cond3)
        elif action.startswith('NT'):
            cond1 = not self.buffer.empty
            cond2 = self.stack.num_open_nonterminals < 100
            return cond1 and cond2
        # TODO: Fix this in the Dictionary class in data.py
        elif action in [PAD_TOKEN, EMPTY_TOKEN, REDUCED_TOKEN]:
            return False
        else:
            raise ValueError('got illegal action: {}'.format(action))
