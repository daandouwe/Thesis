import os

import torch
import torch.nn as nn

from datatypes import Item, Word, Nonterminal, Action
from actions import SHIFT, REDUCE, NT, GEN
from data_test import PAD_INDEX
from glove import load_glove, get_vectors
from embedding import ConvolutionalCharEmbedding
from nn import MLP
from encoder import StackLSTM, HistoryLSTM, BufferLSTM
from parser_test import Parser
from loss import LossCompute

##
from memory import print_memory, get_tensors
from pprint import pprint

from collections import Counter
##

class RNNG(nn.Module):
    """Recurrent Neural Network Grammar model."""
    def __init__(self,
                 dictionary,
                 word_embedding,
                 nonterminal_embedding,
                 action_embedding,
                 buffer_encoder,
                 stack_encoder,
                 history_encoder,
                 action_mlp,
                 nonterminal_mlp,
                 loss_compute,
                 dropout,
                 device):
        self.dictionary = dictionary
        self.device = device

        assert (word_embedding.embedding_dim == action_embedding.embedding_dim == nt_embedding.embedding_dim)
        self.embedding_dim = word_embedding.embedding_dim

        self.word_embedding = word_embedding
        self.nonterminal_embedding= nonterminal_embedding
        self.action_embedding = action_embedding

        self.stack_encoder = stack_encoder
        self.buffer_encoder = buffer_encoder
        self.history_encoder = history_encoder

        # MLP for action classifiction
        self.action_mlp = action_mlp
        # MLP for nonterminal classifiction
        self.nonterminal_mlp = nonterminal_mlp

        self.dropout = nn.Dropout(p=dropout)

        # Loss computation
        self.loss_compute = loss_compute

        self.empty_stack_emb = nn.Parameter(torch.zeros(1, self.embedding_dim, device=self.device))
        self.empty_buffer_emb = nn.Parameter(torch.zeros(1, self.embedding_dim, device=self.device))
        self.empty_history_emb = nn.Parameter(torch.zeros(1, self.embedding_dim, device=self.device))

        self.stack = []
        self.buffer = []
        self.history = []

        self._num_open_nt = 0

    def get_input(self):
        stack, buffer, history = self.get_encoded_input()
        return torch.cat((buffer, history, stack), dim=-1)

    def forward(self, sentence, actions):
        self.initialize(sentence)
        loss = torch.zeros(1, device=self.device)
        for i, action in enumerate(actions):
            # Compute loss
            x = self.get_input()
            action_logits = self.action_mlp(x)
            loss += self.loss_compute(action_logits, action.action_index)
            # If we open a nonterminal, predict which.
            if action.is_nt:
                nonterminal_logits = self.nonterminal_mlp(x)
                nt = action.get_nt()
                loss += self.loss_compute(nonterminal_logits, nt.index)
            self.parse_step(action)
        return loss

    def initialize(self, sentence):
        """Initialize all the components of the parser."""
        self.stack = []
        self.buffer = []
        self.history = []

        self.stack.append(self.empty_stack_emb)
        self.history.append(self.empty_buffer_emb)
        self.buffer.initialize(sentence)
        self.stack.training = self.training

    def _can_shift(self):
        cond1 = not self.buffer.empty
        cond2 = self.stack.num_open_nonterminals > 0
        return cond1 and cond2

    def _can_gen(self):
        # TODO
        return True

    def _can_open(self):
        cond1 = not self.buffer.empty
        cond2 = self.stack.num_open_nonterminals < 100
        return cond1 and cond2

    def _can_reduce(self):
        cond1 = not self.last_action.is_nt
        cond3 = self.stack.num_open_nonterminals > 1
        cond4 = self.buffer.empty
        return (cond1 and cond3) or cond4

    def _shift(self):
        assert self._can_shift(), f'cannot shift: {self}'
        self.stack.push(self.buffer.pop())

    def _gen(self, word):
        assert isinstance(word, Word)
        assert self._can_gen(), f'cannot gen: {self}'
        self.terminals.push(word)

    def _open(self, nonterminal):
        assert isinstance(nonterminal, Nonterminal)
        assert self._can_open(), f'cannot open: {self}'
        self.stack.open(nonterminal)

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
        assert isinstance(action, Action)
        if action == SHIFT:
            self._shift()
        elif action == REDUCE:
            self._reduce()
        elif action.is_gen:
            self._gen(action.get_gen())
        elif action.is_nt:
            self._open(action.get_nt())
        else:
            raise ValueError(f'got illegal action: {action:!r}')
        self.history.push(action)

    def is_valid_action(self, action):
        """Check whether the action is valid under the parser's configuration."""
        assert isinstance(action, Action)
        if action == SHIFT:
            return self._can_shift()
        elif action == REDUCE:
            return self._can_reduce()
        elif action.is_gen:
            return self._can_gen()
        elif action.is_nt:
            return self._can_open()
        else:
            raise ValueError(f'got illegal action: {action:!r}')

    @property
    def actions(self):
        """Return the current history of actions."""
        return self.history.actions

    @property
    def last_action(self):
        """Return the last action taken."""
        return self.history.top

    def make_action(self, index):
        """Maps index to action."""
        assert index in range(3)
        if index == SHIFT.index:
            return SHIFT
        elif index == REDUCE.index:
            return REDUCE
        elif index == Action.NT_INDEX:
            return NT(Nonterminal('_', -1))






def set_embedding(embedding, tensor):
    """Sets the tensor as fixed weight tensor in embedding."""
    assert tensor.shape == embedding.weight.shape
    embedding.weight = nn.Parameter(tensor)
    embedding.weight.requires_grad = False


def make_model(args, dictionary):
    # Embeddings
    num_words = dictionary.num_words
    num_nonterminals = dictionary.num_nonterminals
    num_actions = 3
    if args.use_char:
        word_embedding = ConvolutionalCharEmbedding(
                num_words, args.emb_dim, padding_idx=PAD_INDEX,
                dropout=args.dropout, device=args.device
            )
    else:
        if args.use_fasttext:
            print('FasText only availlable in 300 dimensions: changed word-emb-dim accordingly.')
            args.emb_dim = 300
        word_embedding = nn.Embedding(
                num_words, args.emb_dim, padding_idx=PAD_INDEX
            )
        # Get words in order.
        words = [dictionary.i2w[i] for i in range(num_words)]
        if args.use_glove:
            dim = args.emb_dim
            assert dim in (50, 100, 200, 300), f'invalid dim: {dim}, choose from (50, 100, 200, 300).'
            logfile = open(os.path.join(args.logdir, 'glove.error.txt'), 'w')
            if args.glove_torchtext:
                from torchtext.vocab import GloVe
                print(f'Loading GloVe vectors glove.42B.{args.emb_dim}d (torchtext)...')
                glove = GloVe(name='42B', dim=args.emb_dim)
                embeddings = get_vectors(words, glove, args.emb_dim, logfile)
            else:
                print(f'Loading GloVe vectors glove.6B.{args.emb_dim}d (custom)...')
                embeddings = load_glove(words, args.emb_dim, args.glove_dir, logfile)
            set_embedding(word_embedding, embeddings)
            logfile.close()
        if args.use_fasttext:
            from torchtext.vocab import FastText
            print(f'Loading FastText vectors fasttext.en.300d (torchtext)...')
            fasttext = FastText()
            logfile = open(os.path.join(args.logdir, 'fasttext.error.txt'), 'w')
            embeddings = get_vectors(words, fasttext, args.emb_dim, logfile)
            set_embedding(word_embedding, embeddings)
            logfile.close()

    nonterminal_embedding = nn.Embedding(num_nonterminals, args.emb_dim, padding_idx=PAD_INDEX)
    action_embedding = nn.Embedding(num_actions, args.emb_dim, padding_idx=PAD_INDEX)

    # Encoders
    buffer_encoder = BufferLSTM(
        args.emb_dim,
        args.word_lstm_hidden,
        args.lstm_num_layers,
        args.dropout,
        args.device
    )
    stack_encoder = StackLSTM(
        args.emb_dim,
        args.word_lstm_hidden,
        args.dropout,
        args.device,
        attn_comp=args.use_attn
    )
    history_encoder = HistoryLSTM(
        args.emb_dim,
        args.action_lstm_hidden,
        args.dropout,
        args.device
    )

    # Score MLPs
    mlp_input = 2 * args.word_lstm_hidden + args.action_lstm_hidden
    action_mlp = MLP(
        mlp_input,
        args.mlp_dim,
        num_actions,
        dropout=args.dropout,
        activation='Tanh'
    )
    nonterminal_mlp = MLP(
        mlp_input,
        args.mlp_dim,
        num_nonterminals,
        dropout=args.dropout,
        activation='Tanh'
    )

    loss_compute = LossCompute(nn.CrossEntropyLoss, args.device)

    model = RNNG(
        dictionary=dictionary,
        word_embedding=word_embedding,
        nonterminal_embedding=nonterminal_embedding,
        action_embedding=action_embedding,
        buffer_encoder=buffer_encoder,
        stack_encoder=stack_encoder,
        history_encoder=history_encoder,
        action_mlp=action_mlp,
        nonterminal_mlp=nonterminal_mlp,
        loss_compute=loss_compute,
        dropout=args.dropout,
        device=args.device
    )

    if not args.disable_glorot:
        # Initialize *all* parameters with Glorot. (Overrides custom LSTM init.)
        for param in model.parameters():
            if param.dim() > 1 and param.requires_grad:
                nn.init.xavier_uniform_(param)

    return model
