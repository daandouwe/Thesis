import os
import sys
import itertools
import time
import multiprocessing as mp
from copy import deepcopy
import string
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from datatypes import Item, Word, Nonterminal, Action, Token
from actions import SHIFT, REDUCE, NT, GEN
from data import Corpus, pad, wrap, get_sentences
from parser import DiscParser, GenParser
from embedding import FineTuneEmbedding
from encoder import StackLSTM, HistoryLSTM, BufferLSTM, TerminalLSTM
from composition import BiRecurrentComposition
from nn import MLP, init_lstm
from glove import load_glove, get_vectors
from utils import Timer, write_losses, get_folders, write_args, ceil_div
from loss import AnnealKL
from data_scripts.get_oracle import unkify


############
#   Data   #
############

class Dictionary:
    """A dictionary for stack, buffer, and action symbols."""

    WORD = '+'

    def __init__(self, path, name, use_chars=False):
        self.n2i = dict()  # nonterminals
        self.w2i = dict()  # words
        self.i2n = []
        self.i2w = []
        self.use_chars = use_chars
        self.initialize()
        self.read(path, name)

    def initialize(self):
        self.w2i[self.WORD] = 0
        self.i2w.append(self.WORD)

    def read(self, path, name):
        with open(os.path.join(path, name + '.vocab'), 'r') as f:
            start = len(self.w2i)
            for i, line in enumerate(f, start):
                w = line.rstrip()
                self.w2i[w] = i
                self.i2w.append(w)
        with open(os.path.join(path, name + '.nonterminals'), 'r') as f:
            start = len(self.n2i)
            for i, line in enumerate(f, start):
                s = line.rstrip()
                self.n2i[s] = i
                self.i2n.append(s)

    @property
    def num_words(self):
        return len(self.w2i)

    @property
    def num_nonterminals(self):
        return len(self.n2i)


class PriorData:
    """A dataset with parse configurations."""
    def __init__(self,
                 path,
                 dictionary,
                 model,
                 textline,
                 use_chars=False,
                 max_lines=-1):
        assert textline in ('original', 'lower', 'unked'), textline
        self.dictionary = dictionary
        self.sentences = []
        self.actions = []
        self.use_chars = use_chars
        self.model = model
        self.textline = textline
        self.read(path, max_lines)

    def __str__(self):
        return f'{len(self.sentences):,} sentences'

    def _order(self, new_order):
        self.sentences = [self.sentences[i] for i in new_order]
        self.actions = [self.actions[i] for i in new_order]

    def _get_actions(self, sentence, actions):
        assert all(isinstance(action, str) for action in actions), actions
        assert all(isinstance(word, Word) for word in sentence), sentence
        action_items = []
        token_idx = 0
        for a in actions:
            if a == 'SHIFT':
                if self.model == 'disc':
                    action = Action('SHIFT', Action.SHIFT_INDEX)
                if self.model == 'gen':
                    word = sentence[token_idx]
                    action = GEN(Word(word, self.dictionary.w2i[word.token.processed]))
                    token_idx += 1
            elif a == 'REDUCE':
                action = Action('REDUCE', Action.REDUCE_INDEX)
            elif a.startswith('NT'):
                nt = a[3:-1]
                action = NT(Nonterminal(nt, self.dictionary.n2i[nt]))
            action_items.append(action)
        return action_items

    def read(self, path, max_lines):
        sents = get_sentences(path)  # a list of dictionaries
        for i, sent_dict in enumerate(tqdm(sents, file=sys.stdout)):
            if max_lines > 0 and i >= max_lines:
                break
            sentence = [Word(Token(self.dictionary.WORD, word), 0) for word in sent_dict[self.textline].split()]
            actions = self._get_actions(sentence, sent_dict['actions'])
            self.sentences.append(sentence)
            self.actions.append(actions)
        self.lengths = [len(sent) for sent in self.sentences]

    def order(self):
        old_order = zip(range(len(self.lengths)), self.lengths)
        new_order, _ = zip(*sorted(old_order, key=lambda t: t[1]))
        self._order(new_order)

    def shuffle(self):
        n = len(self.sentences)
        new_order = list(range(0, n))
        np.random.shuffle(new_order)
        self._order(new_order)

    def batches(self, shuffle=False, length_ordered=False):
        n = len(self.sentences)
        if shuffle:
            self.shuffle()
        if length_ordered:
            self.order()
        batches = []
        for i in range(n):
            sentence = self.sentences[i]
            actions = self.actions[i]
            batches.append((sentence, actions))
        return batches


class PriorCorpus:
    """A corpus of three datasets (train, development, and test) and a dictionary."""
    def __init__(self,
                 data_path='../data',
                 model='disc',
                 textline='unked',
                 name='ptb',
                 use_chars=False,
                 max_lines=-1):
        self.dictionary = Dictionary(
            path=os.path.join(data_path, 'vocab', textline),
            name=name,
            use_chars=use_chars)
        self.train = PriorData(
            path=os.path.join(data_path, 'train', name + '.train.oracle'),
            dictionary=self.dictionary,
            model=model,
            textline=textline,
            use_chars=use_chars,
            max_lines=max_lines)
        self.dev = PriorData(
            path=os.path.join(data_path, 'dev', name + '.dev.oracle'),
            dictionary=self.dictionary,
            model=model,
            textline=textline,
            use_chars=use_chars)
        self.test = PriorData(
            path=os.path.join(data_path, 'test', name + '.test.oracle'),
            dictionary=self.dictionary,
            model=model,
            textline=textline,
            use_chars=use_chars)

    def __str__(self):
        items = (
            'Corpus',
             f'vocab size: {self.dictionary.num_words:,}',
             f'train: {str(self.train)}',
             f'dev: {str(self.dev)}',
             f'test: {str(self.test)}',
        )
        return '\n'.join(items)


##############
#   Models   #
##############

def reduced_items(model):
    """Returns a dictionay with the items that were ivolved in the most recent reduce action."""
    items = dict(
        head=model.stack._reduced_head_item,
        children=model.stack._reduced_child_items,
        reduced=model.stack._reduced_embedding,
        attention=model.stack.encoder.composition._attn,
        gate=model.stack.encoder.composition._gate
    )
    return items


def inspect_reduce(model):
    items = reduced_items(model)
    head, children = items['head'], items['children']
    reduced = items['reduced'].squeeze(0).numpy()
    attention = items['attention'].squeeze(0).data.numpy()
    gate = items['gate'].squeeze(0).data.numpy()
    attentive = [f'{child.token} ({attn:.2f})'
        for child, attn in zip(children, attention)]
    print('  ', head.token, '|', ' '.join(attentive), f'[{gate.mean():.2f}]')


class BiRecurrentEncoder(nn.Module):
    """Bidirectional RNN composition function."""
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_first=True, device=None):
        super(BiRecurrentEncoder, self).__init__()
        self.device = device
        self.fwd_rnn = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout)
        self.bwd_rnn = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout)
        self.linear = nn.Linear(2*hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        init_lstm(self.fwd_rnn)
        init_lstm(self.bwd_rnn)
        self.to(device)

    def _reverse(self, tensor):
        idx = [i for i in range(tensor.size(1) - 1, -1, -1)]
        idx = wrap(idx, device=self.device)
        return tensor.index_select(1, idx)

    def forward(self, x):
        assert len(x.shape) == 3
        # Compute bidirectional encoding.
        hf, _ = self.fwd_rnn(x)                 # [batch, seq, hidden_size]
        hb, _ = self.bwd_rnn(self._reverse(x))  # [batch, seq, hidden_size]
        # Select final representation.
        hf = hf[:, -1, :]  # [batch, hidden_size]
        hb = hb[:, -1, :]  # [batch, hidden_size]
        # Concatenate them.
        h = torch.cat((hf, hb), dim=-1)  # [batch, 2*hidden_size]
        return self.dropout(self.relu(self.linear(h)))  # [batch, output_size]


class GatedScorer(nn.Module):
    def __init__(self, state_dim, latent_dim, output_dim, dropout=0.):
        super(GatedScorer, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.gating = nn.Linear(state_dim + latent_dim, latent_dim)
        self.state2latent = nn.Linear(state_dim, latent_dim)
        self.scorer = nn.Linear(latent_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, u, z):
        # Compute `gates` g.
        g = self.sigmoid(self.gating(torch.cat((u, z), dim=-1)))
        # Compute convex combination.
        h = g * self.state2latent(u) + (1 - g) * z
        h = self.dropout(self.relu(h))
        return self.scorer(h)


class BOWTreeDecoder(DiscParser):
    """A tree structured decoder, effectively a DiscRNNG."""
    NUM_ACTIONS = 3
    NUM_WORDS = 1  # trees are (S * (NP * *) (VP *) *)

    def __init__(self, num_words, num_nonterminals, emb_dim, hidden_size,
                 num_layers, latent_dim, dropout, device=None):
        nonterminal_embedding = nn.Embedding(num_nonterminals, emb_dim)
        action_embedding = nn.Embedding(self.NUM_ACTIONS, emb_dim)
        word_embedding = nn.Embedding(self.NUM_WORDS, emb_dim)

        stack_encoder = StackLSTM(
            emb_dim, hidden_size, dropout, device, composition='attention')
        buffer_encoder = BufferLSTM(
            emb_dim, hidden_size, num_layers, dropout, device)
        history_encoder = HistoryLSTM(
            emb_dim, hidden_size, dropout, device)

        super(BOWTreeDecoder, self).__init__(
            word_embedding,
            nonterminal_embedding,
            action_embedding,
            stack_encoder,
            buffer_encoder,
            history_encoder,
            device=device)

        self.device = device
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Parser configuration to transformed z.
        self.gating = nn.Linear(3*hidden_size + latent_dim, latent_dim)
        self.state2latent = nn.Linear(3*hidden_size, latent_dim)
        self.latent2words = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_words))

    def get_input(self):
        stack, buffer, history = self.get_encoded_input()
        return torch.cat((buffer, history, stack), dim=-1)

    def compute_gating(self, u, z):
        # Compute `gates` g.
        g = self.sigmoid(self.gating(torch.cat((u, z), dim=-1)))
        # Compute convex combination.
        zi = g * self.state2latent(u) + (1 - g) * z
        if not self.training:
            # For inspection.
            self.gs.append(g)
            self.us.append(self.state2latent(u))
            self.zs.append(zi)
        return zi

    def forward(self, sentence, actions, z, tree_dropout=0):
        """Transform z through an RNNG decoder."""
        sentence, actions = deepcopy(sentence), deepcopy(actions)
        # Introduce z via the empty embedding (guards).
        # self.stack.empty_emb = nn.Parameter(z)
        # self.buffer.empty_emb = nn.Parameter(z)
        # self.history.empty_emb = nn.Parameter(z)
        # First encoding with the guards, in this case z.
        self.initialize(sentence)
        self.gs = []
        self.zs = []
        self.us = []
        leaf_zs = []  # to save the transformations of z at the leaf
        for i, action in enumerate(actions):
            # The state of the parser just before the shift
            # is will be used to compute the transformed z.
            if action == SHIFT:
                # We let z_i be an (elementwise) convex combination of z and linear(u).
                u = self.get_input()
                # Apply 'tree-dropout' if training.
                if self.training and tree_dropout > 0:
                    if np.random.random() < tree_dropout:
                        zi = z
                    else:
                        zi = compute_gating(u, z)
                else:
                    zi = compute_gating(u, z)
                # Store it.
                leaf_zs.append(self.dropout(zi))
            # elif action == REDUCE:
            #     if not self.training:
            #         inspect_reduce(self)
            # Advance the parser with the action.
            self.parse_step(action)
        tree = self.stack.get_tree(with_tag=False)
        logits = self.latent2words(torch.cat(leaf_zs, dim=0))
        return logits, tree


class RNNTreeDecoder(GenParser):
    """A tree structured decoder, effectively a DiscRNNG."""
    NUM_ACTIONS = 3

    def __init__(self, dictionary, num_words, num_nonterminals, emb_dim, hidden_size,
                 num_layers, latent_dim, dropout, device=None, use_gating=False):
        nonterminal_embedding = nn.Embedding(num_nonterminals, emb_dim)
        action_embedding = nn.Embedding(self.NUM_ACTIONS, emb_dim)
        word_embedding = nn.Embedding(num_words, emb_dim)

        stack_encoder = StackLSTM(
            emb_dim, hidden_size, dropout, device, composition='attention')
        terminal_encoder = TerminalLSTM(
            emb_dim, hidden_size, dropout, device)
        history_encoder = HistoryLSTM(
            emb_dim, hidden_size, dropout, device)

        super(RNNTreeDecoder, self).__init__(
            word_embedding,
            nonterminal_embedding,
            action_embedding,
            stack_encoder,
            terminal_encoder,
            history_encoder,
            device=device)

        self.dictionary = dictionary
        self.device = device
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Transform z to input.
        self.latent2input = nn.Linear(latent_dim, emb_dim)

        # From u (and z) to vocab.
        self.use_gating = use_gating
        if use_gating:
            self.terminal_scorer = GatedScorer(3*hidden_size, latent_dim, num_words, dropout=dropout)
        else:
            self.terminal_scorer = nn.Linear(3*hidden_size, num_words)

    def get_input(self):
        stack, buffer, history = self.get_encoded_input()
        return torch.cat((buffer, history, stack), dim=-1)

    def set_latent(self, z):
        """Introduce z into the RNNG via the empty embedding."""
        h = self.latent2input(z)
        self.stack.empty_emb = nn.Parameter(h)
        self.terminals.empty_emb = nn.Parameter(h)
        self.history.empty_emb = nn.Parameter(h)

    def score_input(self, u, z):
        if self.use_gating:
            return (u, z)
        else:
            return (u,)

    def forward(self, actions, z):
        """Transform z through an RNNG decoder."""
        actions = deepcopy(actions)
        self.set_latent(z)
        # First encoding with the guards, in this case z.
        self.initialize()
        logits = []
        for i, action in enumerate(actions):
            if action.is_gen:
                # Compute word logits.
                u = self.get_input()
                input = self.score_input(u, z)
                logits.append(self.terminal_scorer(*input))
            # Advance the parser with the gold action.
            self.parse_step(action)
        tree = self.stack.get_tree(with_tag=False)
        logits = torch.cat(logits, dim=0)
        return logits, tree

    def predict(self, sentence, actions, z):
        """Return argmax prediction from p(x|t,z)."""
        sentence, actions = deepcopy(sentence), deepcopy(actions)
        self.set_latent(z)
        # First encoding with the guards, in this case z.
        self.initialize()
        pred = []
        for i, action in enumerate(actions):
            if action.is_gen:
                u = self.get_input()
                input = self.score_input(u, z)
                logits = self.terminal_scorer(*input)
                idx = logits.argmax(dim=-1)
                w = self.dictionary.i2w[idx]
                word = Word(Token(w, w), idx)
                action = GEN(word)
                pred.append(word)
            # Advance the parser with the action.
            self.parse_step(action)
        tree = self.stack.get_tree(with_tag=False)
        return pred, tree

    def sample(self, sentence, actions, z):
        """Sample from the observation model p(x|t,z)."""
        sentence, actions = deepcopy(sentence), deepcopy(actions)
        self.set_latent(z)
        # First encoding with the guards, in this case z.
        self.initialize()
        pred = []
        for i, action in enumerate(actions):
            if action.is_gen:
                u = self.get_input()
                input = self.get_input(u, z)
                logits = self.terminal_scorer(*input)
                idx = dist.Categorical(logits=logits).sample().item()
                w = self.dictionary.i2w[idx]
                word = Word(Token(w, w), idx)
                action = GEN(word)
                pred.append(word)
            # Advance the parser with the action.
            self.parse_step(action)
        tree = self.stack.get_tree(with_tag=False)
        return pred, tree


class Inference(nn.Module):
    def __init__(self, num_words, emb_dim, hidden_dim, latent_dim, num_layers, dropout, device=None):
        super(Inference, self).__init__()
        self.embedding = nn.Embedding(num_words, emb_dim)
        self.encoder = BiRecurrentEncoder(emb_dim, hidden_dim, hidden_dim, num_layers, dropout, device=device)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logsigma = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, indices):
        x = self.dropout(self.embedding(indices))
        h = self.encoder(x)
        return self.mu(h), self.logsigma(h)


class BOWTreeVAE(nn.Module):
    def __init__(self, dictionary, emb_dim, hidden_dim, num_layers, latent_dim, dropout, device=None):
        super(BOWTreeVAE, self).__init__()
        self.dictionary = dictionary
        self.num_words = len(dictionary.w2i)
        self.num_nonterminals = len(dictionary.n2i)
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.device = device

        self.generative = BOWTreeDecoder(
            self.num_words, self.num_nonterminals, emb_dim, hidden_dim, num_layers, latent_dim, dropout, device=device)
        self.inference = Inference(
            self.num_words, emb_dim, hidden_dim, latent_dim, num_layers, dropout, device=device)

        self.init_parameters()

    def init_parameters(self):
        """Initialize parameters with Glorot."""
        for param in self.parameters():
            if param.dim() > 1 and param.requires_grad:
                nn.init.xavier_uniform_(param)

    def sample(self, mu, logsigma):
        std = torch.exp(0.5*logsigma)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x):
        return self.inference(x)

    def decode(self, sentence, actions, z, tree_dropout=0):
        return self.generative(sentence, actions, z, tree_dropout=tree_dropout)

    def forward(self, sentence, actions, tree_dropout=0.):
        indices = [self.dictionary.w2i[word.token.processed] for word in sentence]
        x = wrap([indices], self.device)
        mu, logsigma = self.encode(x)
        if self.training:
            z = self.sample(mu, logsigma)
        else:
            z = mu
            # z = self.sample(mu, logsigma)
        x, tree = self.decode(sentence, actions, z, tree_dropout=tree_dropout)
        return x, mu, logsigma, tree

    def kl(self, mu, logsigma):
        return -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())


class RNNTreeVAE(nn.Module):
    def __init__(self, dictionary, emb_dim, hidden_dim, num_layers, latent_dim, dropout, device=None, use_gating=False):
        super(RNNTreeVAE, self).__init__()
        self.dictionary = dictionary
        self.num_words = len(dictionary.w2i)
        self.num_nonterminals = len(dictionary.n2i)
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.device = device

        self.generative = RNNTreeDecoder(
            dictionary, self.num_words, self.num_nonterminals, emb_dim, hidden_dim,
            num_layers, latent_dim, dropout, device=device, use_gating=use_gating)
        self.inference = Inference(
            self.num_words, emb_dim, hidden_dim, latent_dim, num_layers, dropout, device=device)

        self.init_parameters()

    def init_parameters(self):
        """Initialize parameters with Glorot."""
        for param in self.parameters():
            if param.dim() > 1 and param.requires_grad:
                nn.init.xavier_uniform_(param)

    def sample(self, mu, logsigma):
        std = torch.exp(0.5*logsigma)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x):
        return self.inference(x)

    def decode(self, actions, z):
        return self.generative(actions, z)

    def forward(self, sentence, actions):
        indices = [self.dictionary.w2i[word.token.processed] for word in sentence]
        x = wrap([indices], self.device)
        # Use the sentence to predict posterior.
        mu, logsigma = self.encode(x)
        # Sample from posterior.
        z = self.sample(mu, logsigma)
        # Decode with z and the generative tree actions.
        x, tree = self.decode(actions, z)
        return x, mu, logsigma, tree

    def predict(self, sentence, actions, use_mu=False):
        indices = [self.dictionary.w2i[word.token.processed] for word in sentence]
        x = wrap([indices], self.device)
        mu, logsigma = self.encode(x)
        if use_mu:
            z = mu
        else:
            z = self.sample(mu, logsigma)
        return self.generative.predict(sentence, actions, z)

    def kl(self, mu, logsigma):
        return -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())


############
#   Main   #
############

def batchify(data, batch_size):
    batches = [data[i*batch_size:(i+1)*batch_size]
        for i in range(ceil_div(len(data), batch_size))]
    return batches


def insert_leaves(tree, leaves, dummy_word='+'):
    brackets = tree.split(dummy_word)
    assert len(brackets) == (len(leaves) + 1), brackets
    return ''.join(b + l for b, l in zip(brackets[:-1], leaves)) + brackets[-1]


def plot_word_dist(tokens, logits, path, n=10):
    sorted, _ = nn.functional.softmax(logits, dim=-1).sort(descending=True)
    n = 10
    sorted = sorted.squeeze().data.numpy()[:,:n]
    fig, ax = plt.subplots()
    for k in range(min(sorted.shape[0], 10)):
        plt.plot(range(n), sorted[k], label=tokens[k])
    plt.legend(loc='upper right')
    plt.savefig(path)


def plot_heatmap(tokens, array, path):
    longest_word = max(map(len, tokens))
    top_margin = max(longest_word * 0.2 / 9, 0.2)  # this setting seems to work well
    left_margin = max(longest_word * 0.2 / 12, 0.2)  # this setting seems to work well
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(array, cmap='viridis')
    ax.set_yticklabels(tokens, minor=False)
    ax.set_yticks(np.arange(array.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()
    plt.colorbar(heatmap)
    plt.subplots_adjust(left=left_margin, top=1-top_margin)
    plt.savefig(path)


def word_dropout(actions, p, unk_token, unk_id):
    dropout_actions = []
    for action in actions:
        if action.is_gen and np.random.random() < p:
            word = action.get_word()
            word.token = Token(unk_token, unk_token)
            word.index = unk_id
            action = GEN(word)
        dropout_actions.append(action)
    return dropout_actions


def main(args):
    # Set random seeds.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.set_num_threads(mp.cpu_count())

    # Set cuda.
    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Device: {args.device}.')

    # Make output folder structure.
    subdir, logdir, checkdir, outdir = get_folders(args)
    print(f'Output subdirectory: `{subdir}`.')
    print(f'Saving logs to `{logdir}`.')
    print(f'Saving predictions to `{outdir}`.')
    print(f'Saving models to `{checkdir}`.')
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(checkdir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    tensorboard_writer = SummaryWriter(logdir)

    # Save arguments.
    write_args(args, logdir)

    if args.debug:
        args.max_lines = DEBUG_NUM_LINES

    print(f'Loading data from `{args.data}`...')

    # Construct the model.
    if args.observation_model == 'bow':
        # Words not needed: vocab = {*}
        corpus = PriorCorpus(
            data_path=args.data,
            model=args.model,
            textline=args.textline,
            name=args.name,
            use_chars=args.use_chars,
            max_lines=args.max_lines
        )
        model = BOWTreeVAE(corpus.dictionary, args.emb_dim, args.word_lstm_hidden,
            args.lstm_num_layers, args.latent_dim, args.dropout, device=args.device)
        criterion = nn.CrossEntropyLoss(reduction='sum')
    elif args.observation_model == 'rnn':
        # Words are needed.
        corpus = Corpus(
            data_path=args.data,
            model=args.model,
            textline=args.textline,
            name=args.name,
            use_chars=args.use_chars,
            max_lines=args.max_lines
        )
        model = RNNTreeVAE(corpus.dictionary, args.emb_dim, args.word_lstm_hidden,
            args.lstm_num_layers, args.latent_dim, args.dropout, device=args.device, use_gating=args.use_gating)
        criterion = nn.CrossEntropyLoss(reduction='sum')
    elif args.observation_model == 'crf':
        exit('CRF not availlable')

    if args.checkpoint:
        print(f'Loading model from `{args.checkpoint}`.')
        with open(args.checkpoint, 'rb') as f:
            model = torch.load(f, map_location='cpu')

    model.to(args.device)

    train_dataset = corpus.train.batches(shuffle=True)
    dev_dataset = corpus.dev.batches()
    test_dataset = corpus.test.batches()
    print(corpus)

    # Sometimes we don't want to use all data.
    if args.debug:
        print('Debug mode.')
        dev_dataset = dev_dataset[:DEBUG_NUM_LINES]
        test_dataset = test_dataset[:DEBUG_NUM_LINES]
    elif args.max_lines != -1:
        dev_dataset = dev_dataset[:100]
        test_dataset = test_dataset[:100]


    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)

    annealer = AnnealKL(method='logistic', step=2.5e-3, rate=2500)

    num_updates = 0
    logs = dict(elbo=[], loss=[], kl=[], annealed=[], alpha=[])

    def train():
        model.train()
        np.random.shuffle(train_dataset)
        batches = batchify(train_dataset, args.batch_size)
        for i, batch in enumerate(batches, 1):
            if timer.elapsed() > args.max_time:
                break
            nonlocal num_updates
            num_updates += 1

            alpha = annealer.alpha()
            loss = torch.tensor(0.).to(args.device)
            kl = torch.tensor(0.).to(args.device)
            tree_dropout = (1 - alpha) if args.tree_dropout else 0.  # tree dropout is tied to alpha
            for sentence, actions in batch:
                if args.word_dropout > 0:
                    # Word dropout should be on actions, because we use GenRNNG for RNNTreeDecoder.
                    # The actions are the sentence info that are used in the encoders.
                    unk_token, unk_id = 'UNK', corpus.dictionary.w2i['UNK']
                    actions = word_dropout(
                        actions, p=args.word_dropout, unk_token=unk_token, unk_id=unk_id)
                indices = [corpus.dictionary.w2i[word.token.processed] for word in sentence]
                x = wrap([indices], args.device)
                logits, mu, logsigma, _ = model(sentence, actions)
                loss += criterion(logits, x.squeeze(0))
                kl += model.kl(mu, logsigma)
            loss /= args.batch_size
            kl /= args.batch_size
            elbo = loss + alpha * kl

            logs['elbo'].append(elbo.item())
            logs['loss'].append(loss.item())
            logs['kl'].append(kl.item())
            logs['annealed'].append(alpha * kl.item())
            logs['alpha'].append(alpha)

            optimizer.zero_grad()
            elbo.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            if i % args.print_every == 0:
                elbo = np.mean(logs['elbo'][-args.print_every:])
                loss = np.mean(logs['loss'][-args.print_every:])
                kl = np.mean(logs['kl'][-args.print_every:])
                annealed = np.mean(logs['annealed'][-args.print_every:])
                alpha = np.mean(logs['alpha'][-args.print_every:])
                speed = num_updates / timer.elapsed()
                eta = (len(batches) - num_updates) / speed
                tensorboard_writer.add_scalar('train/loss', loss, num_updates)
                tensorboard_writer.add_scalar('train/kl', kl, num_updates)
                tensorboard_writer.add_scalar('train/elbo', elbo, num_updates)
                tensorboard_writer.add_scalar('train/annealed-kl', annealed, num_updates)
                tensorboard_writer.add_scalar('train/alpha', alpha, num_updates)
                print(f'| Step {i:4d} | ELBO {elbo:.3f} | Loss {loss:.3f} | KL {kl:.3f} '
                      f'| alpha {alpha:.3f} | updates/sec {speed:.1f} '
                      f'| elapsed {timer.format_elapsed()} | eta {timer.format(eta)} |')

            if i % args.eval_every == 0:
                eval()

    def sample():
        print('-'*89)
        print('Sampling...')
        if args.observation_model == 'bow':
            sample_bow()
        elif args.observation_model == 'rnn':
            sample_rnn()
        else:
            exit(f'No sampling yet for {args.observation_model}.')
        print('-'*89)

    def eval():
        print('-'*89)
        print('Evaluating...')
        if args.observation_model == 'bow':
            eval_bow()
        elif args.observation_model == 'rnn':
            eval_rnn()
        else:
            exit(f'No sampling yet for {args.observation_model}.')
        print('-'*89)

    def sample_bow():
        model.eval()
        for i, (sentence, actions) in enumerate(test_dataset[:args.max_lines]):
            indices = [corpus.dictionary.w2i[word.token.processed] for word in sentence]
            x = wrap([indices], args.device)
            # Compute logits.
            logits, mu, logsigma, tree = model(sentence, actions)
            predictions = logits.argmax(dim=-1)
            pred = [corpus.dictionary.i2w[i] for i in predictions]
            gold = [corpus.dictionary.i2w[i] for i in indices]
            acc = (np.array(gold) == np.array(pred)).mean()
            print('{:>10}'.format('gold:'), insert_leaves(tree, gold))
            print('{:>10}'.format('argmax:'), insert_leaves(tree, pred), f'[acc {acc:.2f}]')
            # Plotting
            gates = torch.cat(model.generative.gs, dim=0)
            zs = torch.cat(model.generative.zs, dim=0)
            us = torch.cat(model.generative.us, dim=0)
            plot_heatmap(gold, 1-gates.data.numpy(), os.path.join(logdir, f'gates{i}.pdf'))
            plot_heatmap(gold, zs.data.numpy(), os.path.join(logdir, f'zs{i}.pdf'))
            plot_heatmap(gold, us.data.numpy(), os.path.join(logdir, f'us{i}.pdf'))
            plot_heatmap(gold, mu.data.repeat(len(gold), 1).numpy(), os.path.join(logdir, f'mu{i}.pdf'))
            plot_heatmap(gold, logsigma.data.exp().repeat(len(gold), 1).numpy(), os.path.join(logdir, f'sigma{i}.pdf'))
            plot_word_dist(gold, logits, os.path.join(logdir, f'dist{i}.pdf'))
            # Sample x's
            probs = nn.functional.softmax(logits, dim=-1)
            if args.alpha != 1.0:
                probs = nn.functional.softmax(probs.pow(alpha), dim=-1)
            cat = dist.Categorical(logits=logits)
            for _ in range(args.num_samples):
                indices = cat.sample()
                sample = [corpus.dictionary.i2w[k] for k in indices]
                acc = (indices == x).float().mean().item()
                print('{:>10}'.format('sample:'), insert_leaves(tree, sample), f'[acc {acc:.2f}]')
            print('Entropy of p(x|z,t) (mean over sentence):', round(cat.entropy().mean(dim=0).item(), 2))
            print()

    def sample_rnn():
        model.eval()
        for i, (sentence, actions) in enumerate(test_dataset[:args.max_lines]):
            indices = [corpus.dictionary.w2i[word.token.processed] for word in sentence]
            gold = [corpus.dictionary.i2w[i] for i in indices]
            x = wrap([indices], args.device)
            # Get gold tree.
            _, _, _, gold_tree = model(sentence, actions)
            print('{:>10}'.format('gold:'), gold_tree)
            for _ in range(args.num_samples):
                # Get predicted tree.
                pred, pred_tree = model.predict(sentence, actions)
                acc = (np.array(gold) == np.array(pred)).mean()
                print('{:>10}'.format('sample:'), pred_tree, f'[acc {acc:.2f}]')
            print()
        model.train()

    def eval_bow():
        model.eval()
        for i, (sentence, actions) in enumerate(test_dataset[:2]):
            indices = [corpus.dictionary.w2i[word.token.processed] for word in sentence]
            x = wrap([indices], args.device)
            # Compute logits.
            logits, _, _, tree = model(sentence, actions)
            # Decode with argmax.
            predictions = logits.argmax(dim=-1)
            pred = [corpus.dictionary.i2w[i] for i in predictions]
            gold = [corpus.dictionary.i2w[i] for i in indices]
            acc = (np.array(gold) == np.array(pred)).mean()
            sample = [corpus.dictionary.i2w[k] for k in dist.Categorical(logits=logits).sample()]
            print('{:>10}'.format('gold:'), ' '.join(gold))
            print('{:>10}'.format('argmax:'), ' '.join(pred), f'[acc {acc:.2f}]')
            print('{:>10}'.format('sample:'), ' '.join(sample))
            print()
        model.train()

    def eval_rnn():
        model.eval()
        for i, (sentence, actions) in enumerate(test_dataset[:3]):
            indices = [corpus.dictionary.w2i[word.token.processed] for word in sentence]
            gold = [corpus.dictionary.i2w[i] for i in indices]
            x = wrap([indices], args.device)
            # Get gold tree.
            _, _, _, gold_tree = model(sentence, actions)
            print('{:>10}'.format('gold:'), gold_tree)
            for _ in range(3):
                # Get predicted tree.
                pred, pred_tree = model.predict(sentence, actions)
                acc = (np.array(gold) == np.array(pred)).mean()
                print('{:>10}'.format('sample:'), pred_tree, f'[acc {acc:.2f}]')
            print()
        model.train()

    if args.sample_proposals:
        sample()
    else:
        try:
            timer = Timer()
            for epoch in itertools.count(start=1):
                if epoch > args.max_epochs:
                    break
                if timer.elapsed() > args.max_time:
                    break
                train()
                eval()
        except KeyboardInterrupt:
            eval()

        modelpath = os.path.join(checkdir, 'model.pt')
        with open(modelpath, 'wb') as f:
            torch.save(model, f)


if __name__ == '__main__':
    main()
