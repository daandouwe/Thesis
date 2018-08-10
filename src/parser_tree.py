import torch
import torch.nn as nn

from data import (EMPTY_INDEX, REDUCED_INDEX, EMPTY_TOKEN, REDUCED_TOKEN, PAD_TOKEN,
                    ROOT_INDEX, ROOT_TOKEN, Item, Action, wrap, pad)
from tree import Tree

class TransitionBase(nn.Module):
    """A base class for the Stack, Buffer and History."""
    def __init__(self):
        super(TransitionBase, self).__init__()
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
    def __init__(self, word_embedding, nonterminal_embedding, encoder, device):
        """Initialize the Stack.

        Arguments:
            word_embedding (nn.Embedding): embedding function for words.
            nonterminal_embedding (nn.Embedding): embedding function for nonterminals.
            encoder (nn.Module): recurrent encoder.
            device: device on which computation is done (gpu or cpu).
        """
        super(Stack, self).__init__()
        self.word_embedding = word_embedding
        self.nonterminal_embedding = nonterminal_embedding
        self.encoder = encoder
        self.device = device
        self.initialize()

    def __str__(self):
        return f'{type(self).__name__} ({self.num_open_nonterminals} open NTs): {self.tokens}'

    def initialize(self):
        self.tree = Tree()
        self.encoder.initialize_hidden()
        self.push(Item(ROOT_TOKEN, ROOT_INDEX), 'root')

    def open_nonterminal(self, item):
        """Open a new nonterminal in the tree."""
        self.push(item, 'nonterminal')

    def push(self, item, option):
        assert option in ('root', 'nonterminal', 'leaf')
        if option is 'leaf':
            embedding_fn = self.word_embedding
        else:
            embedding_fn = self.nonterminal_embedding
        # Embed the item.
        item.embedding = embedding_fn(wrap([item.index], self.device))
        # Encode the item.
        item.encoding = self.encoder(item.embedding)
        if option is 'root':
            self.tree.make_root(item)
        elif option is 'nonterminal':
            self.tree.open_nonterminal(item)
        elif option is 'leaf':
            self.tree.make_leaf(item)

    def pop(self):
        """Pop items from the stack until first open nonterminal."""
        head, children = self.tree.close_nonterminal()
        sequence_len = len(children)
        # Add nonterminal label to the beginning and end of children
        children = [child.item for child in children]
        children = [head.item] + children + [head.item]
        # Package embeddings as pytorch tensor
        embeddings = [item.embedding.unsqueeze(0) for item in children]
        embeddings = torch.cat(embeddings, 1) # tensor (batch, seq_len, emb_dim)
        return children, embeddings, sequence_len

    def set_reduced_node_embedding(self, embedding):
        """Change the embedding of the node that was just reduced."""
        # Get the last reduced Node from the tree.
        reduced_node = self.tree.last_closed_nonterminal
        # Set the embedding of its underlying Item.
        reduced_node.item.embedding = embedding

    def reset_hidden(self, sequence_len):
        # TODO change _reset_hidden is encoder so we can remove +1
        self.encoder._reset_hidden(sequence_len + 1)

    def encode_reduced_node(self):
        """Compute the new hidden state of the node that was just reduced."""
        # Get the last reduced Node from the tree.
        reduced_node = self.tree.last_closed_nonterminal
        embedding = reduced_node.item.embedding
        reduced_node.item.encoding = self.encoder(embedding)

    @property
    def empty(self):
        """Returns True if the stack is empty."""
        return self.tree.finished

    @property
    def start(self):
        return self.tree.start

    # @property
    # def finished(self):
    #     return self.tree.finished

    @property
    def num_open_nonterminals(self):
        """Return the number of nonterminal nodes in the tree that are not yet closed."""
        return self.tree.num_open_nonterminals

    @property
    def top_item(self):
        """Overide property from baseclass."""
        return self.tree.current_node.item

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
        # return self._items.pop()
        # TODO figure out this mess.
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
    def __init__(self, embedding, encoder, device):
        """Initialize the History.

        Arguments:
            embedding (nn.Embedding): embedding function for actions.
            device: device on which computation is done (gpu or cpu).
        """
        super(History, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.device = device

    def initialize(self):
        """Initialize the history by push the `empty` item."""
        self._reset()
        self.encoder.initialize_hidden()
        self.push(Action(EMPTY_TOKEN, EMPTY_INDEX))

    def push(self, item):
        """Push action index and vector embedding onto history."""
        item.embedding = self.embedding(wrap([item.index], self.device))
        item.encoding = self.encoder(item.embedding)
        self._items.append(item)

    @property
    def actions(self):
        return [item.symbol.token if item.is_nonterminal else item.token
                    for item in self._items[1:]] # First item in self._items is the empty item

class Parser(nn.Module):
    """The parse configuration."""
    def __init__(self, word_embedding, nonterminal_embedding, action_embedding,
                 stack_encoder, buffer_encoder, history_encoder,
                 moves, device=None):
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
        self.SHIFT, self.REDUCE, self.OPEN = moves
        self.stack = Stack(word_embedding, nonterminal_embedding, stack_encoder, device)
        self.buffer = Buffer(word_embedding, buffer_encoder, device)
        self.history = History(action_embedding, history_encoder, device)

    def __str__(self):
        return '\n'.join(('Parser', str(self.stack), str(self.buffer), str(self.history)))

    def initialize(self, items):
        """Initialize all the components of the parser."""
        self.buffer.initialize(items)
        self.stack.initialize()
        self.history.initialize()

    def shift(self):
        """Shift an item from the buffer to the stack."""
        self.stack.push(self.buffer.pop(), 'leaf')

    def get_embedded_input(self):
        """Return the representations of the stack buffer and history."""
        stack = self.stack.top_embedded     # [batch, word_emb_size]
        buffer = self.buffer.top_embedded   # [batch, word_emb_size]
        history = self.history.top_embedded # [batch, action_emb_size]
        return stack, buffer, history

    def get_encoded_input(self):
        """Return the representations of the stack, buffer and history."""
        stack = self.stack.top_encoded      # [batch, word_lstm_hidden]
        buffer = self.buffer.top_encoded    # [batch, word_lstm_hidden]
        history = self.history.top_encoded  # [batch, action_lstm_hidden]
        return stack, buffer, history

    def parse_step(self, action):
        """Updates parser one step give the action."""
        # Take step prescribed by action.
        self.history.push(action)
        if action.index == self.SHIFT:
            self.shift()
        elif action.index == self.OPEN:
            self.stack.open_nonterminal(action.symbol)
        elif action.index == self.REDUCE:
            children, embeddings, sequence_len = self.stack.pop()
            reduced = self.stack.encoder.composition(embeddings)
            self.stack.set_reduced_node_embedding(reduced)
            self.stack.reset_hidden(sequence_len)
            self.stack.encode_reduced_node()
        else:
            raise ValueError(f'got illegal action: {action.token}')

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
    from encoder import BiRecurrentEncoder, StackLSTM, BufferLSTM, HistoryLSTM
    from nn import MLP
    from loss import LossCompute

    # A test sentence.
    tagged_tree = "(S (NP (NN Champagne) (CC and) (NN dessert)) (VP (VBD followed)) (. .))"
    tree = "(S (NP Champagne and dessert) (VP followed) .)"
    sentence = "Champagne and dessert followed .".split()
    actions = [
        'NT(S)',
        'NT(NP)',
        'SHIFT',
        'SHIFT',
        'SHIFT',
        'REDUCE',
        'NT(VP)',
        'SHIFT',
        'REDUCE',
        'SHIFT',
        'REDUCE'
    ]

    def prepare_data(actions, sentence):
        i2a = ['SHIFT', 'REDUCE', 'OPEN']
        i2n = [a[3:-1] for a in actions if a.startswith('NT')] + [ROOT_TOKEN]
        i2w = [w for w in list(set(sentence))]
        a2i = dict((a, i) for i, a in enumerate(i2a))
        n2i = dict((n, i) for i, n in enumerate(i2n))
        w2i = dict((w, i) for i, w in enumerate(i2w))
        action_items = []
        for action in actions:
            if action.startswith('NT'):
                n = action[3:-1]
                symbol = Item(n, n2i[n])
                action = Action('OPEN', a2i['OPEN'], symbol=symbol)
            else:
                action = Action(action, a2i[action])
            action_items.append(action)
        sentence_items = []
        for w in sentence:
            item = Item(w, w2i[w])
            sentence_items.append(item)
        moves = range(3)
        return action_items, sentence_items, moves, len(i2w), len(i2n), len(i2a)

    def test_parser(actions, sentence, dim=4):
        assert dim % 2 == 0
        actions, sentence, moves, num_words, num_nonterm, num_actions = prepare_data(actions, sentence)
        SHIFT, REDUCE, OPEN = moves

        word_embedding = nn.Embedding(num_words, dim)
        nonterminal_embedding = nn.Embedding(num_nonterm, dim)
        action_embedding = nn.Embedding(num_actions, dim)

        stack_encoder = StackLSTM(dim, dim, dropout=0.3)
        buffer_encoder = BufferLSTM(dim, dim, 2, dropout=0.3)
        history_encoder = HistoryLSTM(dim, dim, dropout=0.3)
        reducer = BiRecurrentEncoder(dim, dim//2, 2, dropout=0.3)
        parser = Parser(
            word_embedding,
            nonterminal_embedding,
            action_embedding,
            stack_encoder,
            buffer_encoder,
            history_encoder,
            moves
        )
        parser.initialize(sentence)
        for i, action in enumerate(actions):
            parser.history.push(action)
            print('--------')
            print(f'Step {i:>3}')
            print(f'open nonterminals: {parser.stack.num_open_nonterminals}')
            print('head:', parser.stack.tree.get_current_head())
            print('current:', parser.stack.tree.current_node)
            print('hidden:', parser.stack.encoder.hx1.data)
            # Take step prescribed by action.
            if action.index is SHIFT:
                parser.shift()
            elif action.index is OPEN:
                parser.stack.open_nonterminal(action.symbol)
            elif action.index is REDUCE:
                children, embeddings, sequence_len = parser.stack.pop()
                reduced = reducer(embeddings)
                print(f'reducing: {[item.token for item in children]}')
                print('tensor', embeddings.shape)
                print('tensor', reduced.shape)
                print('{:<17}'.format('reduced:'), reduced.data)
                print('{:<17}'.format('embedding before:'), parser.stack.tree.last_closed_nonterminal.item.embedding.data)
                print('{:<17}'.format('encoding before:'), parser.stack.tree.last_closed_nonterminal.item.encoding.data)
                print('{:<17}'.format('hidden before:'), parser.stack.encoder.hx1.data)
                parser.stack.set_reduced_node_embedding(reduced)
                parser.stack.reset_hidden(sequence_len)
                print('{:<17}'.format('hidden reset:'), parser.stack.encoder.hx1.data)
                parser.stack.encode_reduced_node()
                print('{:<17}'.format('embedding after:'), parser.stack.tree.last_closed_nonterminal.item.embedding.data)
                print('{:<17}'.format('encoding after:'), parser.stack.tree.last_closed_nonterminal.item.encoding.data)
                print('{:<17}'.format('hidden after:'), parser.stack.encoder.hx1.data)
            # Show partial prediction.
            print('partial tree:', parser.stack.tree.linearize())
            print()
        print('--------')
        print('Finished')
        print(f'open nonterminals: {parser.stack.num_open_nonterminals}')
        print('head:', parser.stack.tree.get_current_head())
        print('current:', parser.stack.tree.current_node)
        print()
        print('pred :', parser.stack.tree.linearize())
        print('gold :',tree)

    def forward(model, actions, sentence):
        parser, reducer, action_mlp, nonterminal_mlp = model
        parser.initialize(sentence)
        loss_compute = LossCompute(nn.CrossEntropyLoss, device=None)
        loss = torch.zeros(1)
        for i, action in enumerate(actions):
            # Compute loss
            stack, buffer, history = parser.get_encoded_input()
            x = torch.cat((buffer, history, stack), dim=-1)
            action_logits = action_mlp(x)
            loss += loss_compute(action_logits, action.index)
            # If we open a nonterminal, predict which.
            if action.index is parser.OPEN:
                nonterminal_logits = nonterminal_mlp(x)
                loss += loss_compute(nonterminal_logits, action.symbol.index)

            # Take step prescribed by action.
            parser.history.push(action)
            if action.index is parser.SHIFT:
                parser.shift()
            elif action.index is parser.OPEN:
                parser.stack.open_nonterminal(action.symbol)
            elif action.index is parser.REDUCE:
                children, embeddings, sequence_len = parser.stack.pop()
                reduced = reducer(embeddings)
                parser.stack.set_reduced_node_embedding(reduced)
                parser.stack.reset_hidden(sequence_len)
                parser.stack.encode_reduced_node()
        return loss

    def test_train(actions, sentence, steps, dim=4):
        assert dim % 2 == 0
        data = prepare_data(actions, sentence)
        actions, sentence, moves, num_words, num_nonterm, num_actions = data
        SHIFT, REDUCE, OPEN = moves

        word_embedding = nn.Embedding(num_words, dim)
        nonterminal_embedding = nn.Embedding(num_nonterm, dim)
        action_embedding = nn.Embedding(num_actions, dim)

        stack_encoder = StackLSTM(dim, dim, dropout=0.3)
        buffer_encoder = BufferLSTM(dim, dim, 2, dropout=0.3)
        history_encoder = HistoryLSTM(dim, dim, dropout=0.3)
        parser = Parser(
            word_embedding,
            nonterminal_embedding,
            action_embedding,
            stack_encoder,
            buffer_encoder,
            history_encoder,
            moves
        )
        reducer = BiRecurrentEncoder(dim, dim//2, 2, dropout=0.3)
        action_mlp = MLP(3*dim, dim, num_actions)
        nonterminal_mlp = MLP(3*dim, dim, num_nonterm)

        parameters = list(parser.parameters()) + \
                        list(reducer.parameters()) + \
                        list(action_mlp.parameters()) + \
                        list(nonterminal_mlp.parameters())
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        model = (parser, reducer, action_mlp, nonterminal_mlp)
        for i in range(steps):
            loss = forward(model, actions, sentence)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('loss', loss.item(), end='\r')

    # test_parser(actions, sentence)

    # test_train(actions, sentence, dim=50, steps=1000)
