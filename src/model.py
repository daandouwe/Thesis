import os
from copy import deepcopy

import torch
import torch.nn as nn

from datatypes import Item, Word, Nonterminal, Action
from actions import SHIFT, REDUCE, NT, GEN
from data import PAD_INDEX
from glove import load_glove, get_vectors
from embedding import ConvolutionalCharEmbedding
from nn import MLP
from encoder import LATENT_COMPOSITIONS, StackLSTM, HistoryLSTM, BufferLSTM, TerminalLSTM
from parser import DiscParser, GenParser
from loss import LossCompute, ElboCompute


class DiscRNNG(DiscParser):
    """Discriminative Recurrent Neural Network Grammar."""
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
                 criterion,
                 dropout,
                 device):
        super(DiscRNNG, self).__init__(
            word_embedding,
            nonterminal_embedding,
            action_embedding,
            stack_encoder,
            buffer_encoder,
            history_encoder,
            device=device
        )
        self.dictionary = dictionary
        self.device = device

        self.composition_type = stack_encoder.composition_type
        self.elbo_objective = self.stack.encoder.requires_kl

        # MLP for action classifiction
        self.action_mlp = action_mlp
        # MLP for nonterminal classifiction
        self.nonterminal_mlp = nonterminal_mlp

        self.dropout = nn.Dropout(p=dropout)

        # Loss computation
        self.criterion = criterion
        # Update counter
        self.i = 0

    def get_input(self):
        stack, buffer, history = self.get_encoded_input()
        return torch.cat((buffer, history, stack), dim=-1)

    def forward(self, sentence, actions):
        """Forward pass only used for training."""
        # We change the items in sentence and actions in-place,
        # so we make a copy so that the tensors do not hang around.
        sentence, actions = deepcopy(sentence), deepcopy(actions)  # Do not remove!
        self.initialize(sentence)
        self.i += 1
        for i, action in enumerate(actions):
            # Compute loss
            x = self.get_input()
            action_logits = self.action_mlp(x)
            self.criterion(action_logits, action.action_index)
            # If we open a nonterminal, predict which.
            if action.is_nt:
                nonterminal_logits = self.nonterminal_mlp(x)
                nt = action.get_nt()
                self.criterion(nonterminal_logits, nt.index)
            self.parse_step(action)
            # Add KL if we use latent factor encoder.
            if action == REDUCE and self.elbo_objective:
                alpha = self.stack.encoder.composition._alpha
                kl = self.stack.encoder.composition.kl(alpha)
                self.criterion.add_kl(kl)
        loss = self.criterion.get_loss(self.i)
        return loss

    def reduced_items(self):
        items = dict()
        items['head'] = self.stack._reduced_head_item
        items['children'] = self.stack._reduced_child_items
        if self.composition_type == 'attention':
            items['attention'] = self.stack.encoder.composition._attn
            items['gate'] = self.stack.encoder.composition._gate
        elif self.composition_type == 'latent-factors':
            items['sample'] = self.model.stack.encoder.composition._sample
            items['alpha'] = self.model.stack.encoder.composition._alpha
        return items


class GenRNNG(GenParser):
    """Generative Recurrent Neural Network Grammar."""
    def __init__(self,
                 dictionary,
                 word_embedding,
                 nonterminal_embedding,
                 action_embedding,
                 terminal_encoder,
                 stack_encoder,
                 history_encoder,
                 action_mlp,
                 nonterminal_mlp,
                 terminal_mlp,
                 criterion,
                 dropout,
                 device):
        super(GenRNNG, self).__init__(
            word_embedding,
            nonterminal_embedding,
            action_embedding,
            stack_encoder,
            terminal_encoder,
            history_encoder,
            device=device
        )
        self.dictionary = dictionary
        self.device = device

        # MLP for actions.
        self.action_mlp = action_mlp
        # MLP for nonterminals.
        self.nonterminal_mlp = nonterminal_mlp
        # MLP for words.
        self.terminal_mlp = terminal_mlp

        self.dropout = nn.Dropout(p=dropout)

        # Loss computation
        self.criterion = criterion

    def get_input(self):
        stack, terminals, history = self.get_encoded_input()
        return torch.cat((terminals, history, stack), dim=-1)

    def forward(self, sentence, actions):
        """Forward pass only used for training."""
        # We change the items in sentence and actions in-place,
        # so we make a copy so that the tensors do not hang around.
        sentence, actions = deepcopy(sentence), deepcopy(actions)  # Do not remove!
        self.initialize(sentence)
        loss = torch.zeros(1, device=self.device)
        for i, action in enumerate(actions):
            # Compute loss
            x = self.get_input()
            action_logits = self.action_mlp(x)
            loss += self.criterion(action_logits, action.action_index)
            # If we open a nonterminal, predict which.
            if action.is_nt:
                nonterminal_logits = self.nonterminal_mlp(x)
                nt = action.get_nt()
                loss += self.criterion(nonterminal_logits, nt.index)
            # If we generate a word, predict which.
            if action.is_gen:
                terminal_logits = self.terminal_mlp(x)
                word = action.get_word()
                loss += self.criterion(terminal_logits, word.index)
            self.parse_step(action)
        return loss


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
    if args.use_chars:
        word_embedding = ConvolutionalCharEmbedding(
                num_words,
                args.emb_dim,
                padding_idx=PAD_INDEX,
                dropout=args.dropout,
                device=args.device
            )
    else:
        if args.use_fasttext:
            print('FasText only availlable in 300 dimensions: changed word-emb-dim accordingly.')
            args.emb_dim = 300
        word_embedding = nn.Embedding(
                num_words, args.emb_dim, padding_idx=PAD_INDEX)
        # Get words in order.
        words = [dictionary.i2w[i] for i in range(num_words)]
        if args.use_glove:
            dim = args.emb_dim
            assert dim in (50, 100, 200, 300), f'invalid dim: {dim}, choose from (50, 100, 200, 300).'
            logfile = open(os.path.join(args.logdir, 'glove.error.txt'), 'w')
            if args.glove_torchtext:
                from torchtext.vocab import GloVe
                print(f'Loading GloVe vectors `glove.42B.{args.emb_dim}d` (torchtext loader)...')
                glove = GloVe(name='42B', dim=args.emb_dim)
                embeddings = get_vectors(words, glove, args.emb_dim, logfile)
            else:
                print(f'Loading GloVe vectors `glove.6B.{args.emb_dim}d` (custom loader)...')
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
    terminal_encoder = TerminalLSTM(
        args.emb_dim,
        args.word_lstm_hidden,
        args.dropout,
        args.device
    )
    stack_encoder = StackLSTM(
        args.emb_dim,
        args.word_lstm_hidden,
        args.dropout,
        args.device,
        composition=args.composition
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
        activation=args.mlp_nonlinearity
    )
    nonterminal_mlp = MLP(
        mlp_input,
        args.mlp_dim,
        num_nonterminals,
        dropout=args.dropout,
        activation=args.mlp_nonlinearity
    )
    terminal_mlp = MLP(
        mlp_input,
        args.mlp_dim,
        num_words,
        dropout=args.dropout,
        activation=args.mlp_nonlinearity
    )

    elbo_objective = (args.composition in LATENT_COMPOSITIONS)
    if elbo_objective:
        criterion = ElboCompute(nn.CrossEntropyLoss, args.device, anneal=args.disable_kl_anneal)
    else:
        criterion = LossCompute(nn.CrossEntropyLoss, args.device)
    if args.model == 'disc':
        model = DiscRNNG(
            dictionary=dictionary,
            word_embedding=word_embedding,
            nonterminal_embedding=nonterminal_embedding,
            action_embedding=action_embedding,
            buffer_encoder=buffer_encoder,
            stack_encoder=stack_encoder,
            history_encoder=history_encoder,
            action_mlp=action_mlp,
            nonterminal_mlp=nonterminal_mlp,
            criterion=criterion,
            dropout=args.dropout,
            device=args.device
        )
    if args.model == 'gen':
        model = GenRNNG(
            dictionary=dictionary,
            word_embedding=word_embedding,
            nonterminal_embedding=nonterminal_embedding,
            action_embedding=action_embedding,
            terminal_encoder=terminal_encoder,
            stack_encoder=stack_encoder,
            history_encoder=history_encoder,
            action_mlp=action_mlp,
            nonterminal_mlp=nonterminal_mlp,
            terminal_mlp=terminal_mlp,
            criterion=criterion,
            dropout=args.dropout,
            device=args.device
        )

    if not args.disable_glorot:
        # Initialize *all* parameters with Glorot. (Overrides custom LSTM init.)
        for param in model.parameters():
            if param.dim() > 1 and param.requires_grad:
                nn.init.xavier_uniform_(param)

    return model
