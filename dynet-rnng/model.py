import dynet as dy
import numpy as np

from parser import DiscParser, GenParser, Stack, Buffer, History, Terminal
from embedding import Embedding, FineTuneEmbedding, PretrainedEmbedding
from encoder import StackLSTM
from composition import AttentionComposition, BiRecurrentComposition
from nn import MLP


class DiscRNNG(DiscParser):
    """Discriminative Recurrent Neural Network Grammar."""

    def __init__(
        self,
        model,
        dictionary,
        num_words,
        num_nt,
        word_emb_dim,
        nt_emb_dim,
        action_emb_dim,
        stack_hidden_size,
        buffer_hidden_size,
        history_hidden_size,
        stack_num_layers,
        buffer_num_layers,
        history_num_layers,
        composition,
        mlp_hidden,
        dropout,
        use_glove=False,
        glove_dir=None,
        fine_tune_embeddings=False,
        freeze_embeddings=False
    ):
        assert composition in ('basic', 'attention'), composition
        self.num_words = num_words
        self.num_nt = num_nt
        self.num_actions = 2 + num_nt

        self.word_emb_dim = word_emb_dim
        self.nt_emb_dim = nt_emb_dim
        self.action_emb_dim = action_emb_dim

        self.dictionary = dictionary

        # Embeddings
        self.nt_embedding = Embedding(model, num_nt, nt_emb_dim)
        self.action_embedding = Embedding(model, self.num_actions, action_emb_dim)
        if use_glove:
            if fine_tune_embeddings:
                self.word_embedding = FineTuneEmbedding(
                    model, num_words, word_emb_dim, glove_dir, dictionary.i2w)
            else:
                self.word_embedding = PretrainedEmbedding(
                    model, num_words, word_emb_dim, glove_dir, dictionary.i2w, freeze=freeze_embeddings)
        else:
            self.word_embedding = Embedding(model, num_words, word_emb_dim)

        # self.word_embedding = model.add_lookup_parameters((num_words, word_emb_dim))
        # self.nt_embedding = model.add_lookup_parameters((num_nt, nt_emb_dim))
        # self.action_embedding = model.add_lookup_parameters((self.num_actions, action_emb_dim))

        # Encoders
        self.stack_encoder = StackLSTM(
            model, word_emb_dim, stack_hidden_size, stack_num_layers, dropout)
        self.buffer_encoder = StackLSTM(
            model, word_emb_dim, buffer_hidden_size, buffer_num_layers, dropout)
        self.history_encoder = StackLSTM(
            model, action_emb_dim, history_hidden_size, history_num_layers, dropout)

        # Composition function
        if composition == 'basic':
            self.composer = BiRecurrentComposition(model, word_emb_dim, stack_num_layers, dropout)
        elif composition == 'attention':
            self.composer = AttentionComposition(model, word_emb_dim, stack_num_layers, dropout)

        # Transition system
        self.stack = Stack(
            model, dictionary, self.word_embedding, self.nt_embedding, self.stack_encoder, self.composer)
        self.buffer = Buffer(
            model, dictionary, self.word_embedding, self.buffer_encoder)
        self.history = History(
            model, dictionary, self.action_embedding, self.history_encoder)

        # Scorers
        parse_repr_dim = stack_hidden_size + buffer_hidden_size + history_hidden_size
        self.action_mlp = MLP(model, parse_repr_dim, mlp_hidden, self.num_actions)

    def __call__(self, words, actions):
        """Forward pass for training."""
        self.initialize(words)
        nll = 0.0
        for action_id in actions:
            # This gets you the string, e.g. `NT(S)` or `REDUCE`.
            action = self.dictionary.i2a[action_id]
            # Compute action loss
            u = self.parser_representation()
            action_logits = self.action_mlp(u)
            nll += dy.pickneglogsoftmax(action_logits, action_id)
            # Move the parser ahead.
            self.parse_step(action, action_id)
        return nll

    def parse(self, words):
        """Greedy decoding for prediction."""
        nll = 0.0
        self.initialize(words)
        while not self.stack.is_finished():
            u = self.parser_representation()
            action_logits = self.action_mlp(u)
            mask = self.valid_actions_mask()
            allowed_logits = action_logits + mask
            action_id = np.argmax(allowed_logits.value())
            nll += dy.pickneglogsoftmax(action_logits, action_id)
            action = self.dictionary.i2a[action_id]
            self.parse_step(action, action_id)
        return self.stack.get_tree(), nll

    def valid_actions_mask(self):
        """Mask invallid actions for decoding."""
        mask = np.zeros(self.num_actions)
        for i in range(self.num_actions):
            if self.is_valid_action(self.dictionary.i2a[i]):
                mask[i] = 0.
            else:
                mask[i] = -np.inf
        return dy.inputTensor(mask)


class GenRNNG(GenParser):
    """Discriminative Recurrent Neural Network Grammar."""

    def __init__(
        self,
        model,
        dictionary,
        num_words,
        num_nt,
        word_emb_dim,
        nt_emb_dim,
        action_emb_dim,
        stack_hidden_size,
        terminal_hidden_size,
        history_hidden_size,
        stack_num_layers,
        terminal_num_layers,
        history_num_layers,
        composition,
        mlp_hidden,
        dropout,
    ):
        assert composition in ('basic', 'attention'), composition
        self.num_words = num_words
        self.num_nt = num_nt
        self.num_actions = 1 + num_nt + num_words

        self.word_emb_dim = word_emb_dim
        self.nt_emb_dim = nt_emb_dim
        self.action_emb_dim = action_emb_dim

        self.dictionary = dictionary

        # Embeddings
        self.word_embedding = model.add_lookup_parameters((num_words, word_emb_dim))
        self.nt_embedding = model.add_lookup_parameters((num_nt, nt_emb_dim))
        self.action_embedding = model.add_lookup_parameters((self.num_actions, action_emb_dim))

        # Encoders
        self.stack_encoder = StackLSTM(
            model, word_emb_dim, stack_hidden_size, stack_num_layers, dropout)
        self.terminal_encoder = StackLSTM(
            model, word_emb_dim, terminal_hidden_size, terminal_num_layers, dropout)
        self.history_encoder = StackLSTM(
            model, action_emb_dim, history_hidden_size, history_num_layers, dropout)

        # Composition function
        if composition == 'basic':
            self.composer = BiRecurrentComposition(model, word_emb_dim, stack_num_layers, dropout)
        elif composition == 'attention':
            self.composer = AttentionComposition(model, word_emb_dim, stack_num_layers, dropout)

        # Transition system
        self.stack = Stack(
            model, dictionary, self.word_embedding, self.nt_embedding, self.stack_encoder, self.composer)
        self.terminal = Terminal(
            model, dictionary, self.word_embedding, self.terminal_encoder)
        self.history = History(
            model, dictionary, self.action_embedding, self.history_encoder)

        # Scorers
        parse_repr_dim = stack_hidden_size + terminal_hidden_size + history_hidden_size
        self.action_mlp = MLP(model, parse_repr_dim, mlp_hidden, 3)  # REDUCE, NT, GEN
        self.nt_mlp = MLP(model, parse_repr_dim, mlp_hidden, self.num_nt)  # NT(S), NT(NP), ...
        self.word_mlp = MLP(model, parse_repr_dim, mlp_hidden, self.num_words)  # GEN(the), GEN(cat), ...

    def __call__(self, words, actions):
        """Forward pass for training."""
        self.initialize()
        nll = 0.
        for action_id in actions:
            # Compute action loss
            u = self.parser_representation()
            action_logits = self.action_mlp(u)
            nll += dy.pickneglogsoftmax(action_logits, self._get_action_id(action_id))
            if self._is_nt_id(action_id):
                nt_logits = self.nt_mlp(u)
                nll += dy.pickneglogsoftmax(nt_logits, self._get_nt_id(action_id))
            elif self._is_gen_id(action_id):
                word_logits = self.word_mlp(u)
                nll += dy.pickneglogsoftmax(word_logits, self._get_word_id(action_id))
            # Move the parser ahead.
            self.parse_step(self.dictionary.i2a[action_id], action_id)
        return nll

    def valid_actions_mask(self):
        """Mask invallid actions for decoding."""
        mask = np.zeros(self.num_actions)
        for i in range(self.num_actions):
            if self.is_valid_action(self.dictionary.i2a[i]):
                mask[i] = 0.
            else:
                mask[i] = -np.inf
        return dy.inputTensor(mask)
