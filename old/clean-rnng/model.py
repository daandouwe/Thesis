"""
1. DiscRNNG inherits from DiscParser, GenRNNG inherits from GenParser.
2. We keep model.forward(self, sentence, actions) but now
   both sentence and actions are isntances of `torch.LongTensor`.
   This will allow use to use Matchbox's `@matchbox.batch`
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import matchbox
# import matchbox.functional as F

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
        dictionary,
        num_words,
        num_nt,
        word_emb_dim,
        nt_emb_dim,
        action_emb_dim,
        stack_emb_dim,
        buffer_emb_dim,
        history_emb_dim,
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
        self.word_embedding = nn.Embedding(num_words, word_emb_dim)
        self.nt_embedding = nn.Embedding(num_nt, nt_emb_dim)
        self.action_embedding = nn.Embedding(self.num_actions, action_emb_dim)

        # Encoders
        self.stack_encoder = StackLSTM(
            stack_emb_dim, stack_hidden_size, stack_num_layers, dropout, device)
        self.buffer_encoder = StackLSTM(
            buffer_emb_dim, buffer_hidden_size, buffer_num_layers, dropout, device)
        self.history_encoder = StackLSTM(
            history_emb_dim, history_hidden_size, history_num_layers, dropout, device)

        # Composition function
        self.composer = BiRecurrentComposition(stack_emb_dim, stack_num_layers, dropout, device)
        # self.composer = AttentionComposition(stack_emb_dim, stack_num_layers, dropout, device)

        # Transition system
        self.stack = Stack(
            dictionary, self.word_embedding, self.nt_embedding, self.stack_encoder, self.composer, device)
        self.buffer = Buffer(
            dictionary, self.word_embedding, self.buffer_encoder, device)
        self.history = History(
            dictionary, self.action_embedding, self.history_encoder, device)

        # Scorers
        parse_repr_dim = stack_hidden_size + buffer_hidden_size + history_hidden_size
        self.action_mlp = MLP(parse_repr_dim, mlp_hidden, self.num_actions)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words: Tensor, actions: Tensor):
        """Forward pass only used for training."""
        # words: [1, seq_len]
        # actions: [1, seq_len]
        self.initialize(words)
        llh = 0.
        for i, action_id in enumerate(actions.unbind(1)):
            print(self.state())
            # This get's you the string, e.g. `NT(S)` or `REDUCE`.
            action = self.dictionary.i2a[action_id.item()]
            # Compute action loss
            u = self.parser_representation()
            action_logprobs = F.log_softmax(self.action_mlp(u))
            llh += action_logprobs[:, action_id]
            # Move the parser ahead.
            self.parse_step(action, action_id)
        print(self.state())
        return llh
