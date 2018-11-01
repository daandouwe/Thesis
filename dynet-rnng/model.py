import dynet as dy
import numpy as np

from parser import DiscParser, Stack, Buffer, History
from encoder import StackLSTM
from composition import AttentionComposition, BiRecurrentComposition
from nn import MLP


class DiscRNNG(DiscParser):
    """Discriminative Recurrent Neural Network Grammar."""
    SHIFT_ID = 0
    REDUCE_ID = 1

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
        mlp_hidden,
        dropout,
        device
    ):
        super(DiscRNNG, self).__init__()
        self.num_words = num_words
        self.num_nt = num_nt
        self.num_actions = num_nt + 2

        self.word_emb_dim = word_emb_dim
        self.nt_emb_dim = nt_emb_dim
        self.action_emb_dim = action_emb_dim

        self.dictionary = dictionary
        self.device = device

        # Embeddings
        self.word_embedding = model.add_lookup_parameters((num_words, word_emb_dim))
        self.nt_embedding = model.add_lookup_parameters((num_nt, nt_emb_dim))
        self.action_embedding = model.add_lookup_parameters((self.num_actions, action_emb_dim))

        # Encoders
        self.stack_encoder = StackLSTM(
            model, word_emb_dim, stack_hidden_size, stack_num_layers, dropout)
        self.buffer_encoder = StackLSTM(
            model, word_emb_dim, buffer_hidden_size, buffer_num_layers, dropout)
        self.history_encoder = StackLSTM(
            model, action_emb_dim, history_hidden_size, history_num_layers, dropout)

        # Composition function
        # self.composer = BiRecurrentComposition(model, word_emb_dim, stack_num_layers, dropout)
        self.composer = AttentionComposition(model, word_emb_dim, stack_num_layers, dropout)

        # Transition system
        self.stack = Stack(
            model, dictionary, self.word_embedding, self.nt_embedding, self.stack_encoder, self.composer, device)
        self.buffer = Buffer(
            model, dictionary, self.word_embedding, self.buffer_encoder, device)
        self.history = History(
            model, dictionary, self.action_embedding, self.history_encoder, device)

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
