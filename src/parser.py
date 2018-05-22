import torch

from newdata import EMPTY_INDEX, wrap

class Stack:
    """The stack"""
    def __init__(self, dictionary, embedding):
        """Initialize the Stack.

        Args:
            dictionary: an instance of data.Dictionary
        """
        self._tokens = [] # list of indices
        self._embeddings = [] # list of embeddings (pytorch vectors)

        self._open_nonterminal = int() # The index of the last opened nonterminal
        self._len_new_bracket = int() #

        self.dict = dictionary
        self.embedding = embedding

    def __str__(self):
        stack = [self.dict.i2s[i] for i in self._tokens]
        return 'Stack : {}'.format(stack)

    def _reset(self):
        """Resets the buffer to empty state."""
        self._tokens = []
        self._embeddings = []

    def initialize(self):
        self._reset()
        self.push(EMPTY_INDEX)

    def pop(self):
        """Pop tokens and vectors from the stack until first open nonterminal."""
        # We pop items from self._tokens till we find a nonterminal.
        found_nonterminal = False
        tokens, embeddings = [], []
        while not found_nonterminal:
            i = self._tokens.pop()
            e = self._embeddings.pop()
            tokens.append(i)
            embeddings.append(e)
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
        return tokens, embeddings

    def push(self, token, vec=None):
        self._tokens.append(token)
        if vec is None:
            vec = self.embedding(wrap([token]))
        self._embeddings.append(vec)

    @property
    def top_embedded(self):
        """Returns the embedding of the symbol on the top of the stack."""
        return self._embeddings[-1]


class Buffer:
    """The buffer."""
    def __init__(self, dictionary, embedding):
        self._tokens = []
        self._embeddings = []

        self.dict = dictionary
        self.embedding = embedding

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
        token = self._tokens.pop()
        vec = self._embeddings.pop()
        # If popping makes the buffer empty, push
        # the empty token to signal that it is empty.
        if not self._tokens:
            self.push(EMPTY_INDEX)
        return token, vec

    def push(self, token):
        """Push action index and vector embedding onto buffer."""
        self._tokens.append(token)
        vec = self.embedding(wrap([token]))
        self._embeddings.append(vec)

    def __str__(self):
        buffer = [self.dict.i2w[i] for i in self._tokens]
        return 'Buffer : {}'.format(buffer)

    @property
    def embedded(self):
        """Concatenate all the embeddings and return as pytorch tensor"""
        embs = [emb.unsqueeze(0) for emb in self._embeddings]
        return torch.cat(embs, 1) # [batch, seq_len, emb_dim]

class History:
    def __init__(self, dictionary, embedding):
        self._actions = []
        self._embeddings = []

        self.dict = dictionary
        self.embedding = embedding

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

    def __str__(self):
        history = [self.dict.i2a[i] for i in self._actions]
        return 'History : {}'.format(history)

    @property
    def embedded(self):
        """Concatenate all the embeddings and return as pytorch tensor"""
        embs = [emb.unsqueeze(0) for emb in self._embeddings]
        return torch.cat(embs, 1) # [batch, seq_len, emb_dim]


class Parser:
    """The parse configuration."""
    def __init__(self, dictionary, embedding, history_embedding):
        self.stack = Stack(dictionary, embedding)
        self.buffer = Buffer(dictionary, embedding)
        self.history = History(dictionary, history_embedding)
        self.dict = dictionary

    def initialize(self, sentence):
        self.stack.initialize()
        self.buffer.initialize(sentence)
        self.history.initialize()

    def shift(self):
        idx, _ = self.buffer.pop()
        # translate between dictionaries
        token = self.dict.i2w[idx]
        idx = self.dict.s2i[token]
        self.stack.push(idx)

    def get_embedded_input(self):
        stack = self.stack.top_embedded # the input on top [batch, emb_size]
        buffer = self.buffer.embedded # the entire buffer [batch, seq_len, emb_size]
        history = self.history.embedded # the entire stack [batch, seq_len, emb_size]
        return stack, buffer, history

    def __str__(self):
        return 'PARSER STATE\n{}\n{}\n{}'.format(self.stack, self.buffer, self.history)

if __name__ == '__main__':
    from data import Dictionary

    dictionary = Dictionary("../tmp/ptb")
