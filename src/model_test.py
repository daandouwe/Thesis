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


class RNNG(Parser):
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
        super(RNNG, self).__init__(
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

        # MLP for action classifiction
        self.action_mlp = action_mlp
        # MLP for nonterminal classifiction
        self.nonterminal_mlp = nonterminal_mlp

        self.dropout = nn.Dropout(p=dropout)

        # Loss computation
        self.loss_compute = loss_compute

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

    def parse(self, sentence):
        assert not self.training, f'set model.eval() to enable parsing'
        self.initialize(sentence)
        t = 0
        while not self.stack.is_empty():
            t += 1
            # Compute loss
            x = self.get_input()
            action_logits = self.action_mlp(x)

            # TODO something like mask = (1, -inf, 1) = self.get_illegal_actions()
            # (1.3, 0.2, -inf) = (action_logits * mask).sort(descending=True)
            # action_index = ids[0]

            # Get highest scoring valid predictions.
            vals, ids = action_logits.sort(descending=True)
            vals, ids = vals.data.squeeze(0), ids.data.squeeze(0)
            i = 0
            action = self.make_action(ids[i])
            while not self.is_valid_action(action):
                i += 1
                action = self.make_action(ids[i])
            if action.is_nt:
                nonterminal_logits = self.nonterminal_mlp(x)
                vals, ids = nonterminal_logits.sort(descending=True)
                vals, ids = vals.data.squeeze(0), ids.data.squeeze(0)
                X = Nonterminal(self.dictionary.i2n[ids[0]], ids[0])
                action = NT(X)
            self.parse_step(action)
        return self.stack.get_tree()

    def make_action(self, index):
        """Maps index to action."""
        assert index in range(3)
        if index == SHIFT.index:
            return SHIFT
        elif index == REDUCE.index:
            return REDUCE
        elif index == Action.NT_INDEX:
            return NT(Nonterminal('_', -1))
        # elif index == Action.GEN_INDEX:
            # return GEN(Word('_', -1))


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
                num_words, args.word_emb_dim, padding_idx=PAD_INDEX,
                dropout=args.dropout, device=args.device
            )
    else:
        if args.use_fasttext:
            print('FasText only availlable in 300 dimensions: changed word-emb-dim accordingly.')
            args.word_emb_dim = 300
        word_embedding = nn.Embedding(
                num_words, args.word_emb_dim, padding_idx=PAD_INDEX
            )
        # Get words in order.
        words = [dictionary.i2w[i] for i in range(len(dictionary.w2i))]
        if args.use_glove:
            dim = args.word_emb_dim
            assert dim in (50, 100, 200, 300), f'invalid dim: {dim}, choose from (50, 100, 200, 300).'
            logfile = open(os.path.join(args.logdir, 'glove.error.txt'), 'w')
            if args.glove_torchtext:
                from torchtext.vocab import GloVe
                print(f'Loading GloVe vectors glove.42B.{args.word_emb_dim}d (torchtext)...')
                glove = GloVe(name='42B', dim=args.word_emb_dim)
                embeddings = get_vectors(words, glove, args.word_emb_dim, logfile)
            else:
                print(f'Loading GloVe vectors glove.6B.{args.word_emb_dim}d (custom)...')
                embeddings = load_glove(words, args.word_emb_dim, args.glove_dir, logfile)
            set_embedding(word_embedding, embeddings)
            logfile.close()
        if args.use_fasttext:
            from torchtext.vocab import FastText
            print(f'Loading FastText vectors fasttext.en.300d (torchtext)...')
            fasttext = FastText()
            logfile = open(os.path.join(args.logdir, 'fasttext.error.txt'), 'w')
            embeddings = get_vectors(words, fasttext, args.word_emb_dim, logfile)
            set_embedding(word_embedding, embeddings)
            logfile.close()

    nonterminal_embedding = nn.Embedding(num_nonterminals, args.word_emb_dim, padding_idx=PAD_INDEX)
    action_embedding = nn.Embedding(num_actions, args.word_emb_dim, padding_idx=PAD_INDEX)

    # Encoders
    buffer_encoder = BufferLSTM(
        args.word_emb_dim,
        args.word_lstm_hidden,
        args.lstm_num_layers,
        args.dropout,
        args.device
    )
    stack_encoder = StackLSTM(
        args.word_emb_dim,
        args.word_lstm_hidden,
        args.dropout,
        args.device
    )
    history_encoder = HistoryLSTM(
        args.word_emb_dim,
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
