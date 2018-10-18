import os
import sys
import itertools
import multiprocessing as mp
from copy import deepcopy
import string
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from tensorboardX import SummaryWriter

from datatypes import Item, Word, Nonterminal, Action, Token
from actions import SHIFT, REDUCE, NT, GEN
from data import pad, wrap, get_sentences
from parser import DiscParser
from embedding import FineTuneEmbedding
from encoder import StackLSTM, HistoryLSTM, BufferLSTM, TerminalLSTM
from composition import BiRecurrentComposition
from nn import MLP, init_lstm
from glove import load_glove, get_vectors
from utils import Timer, write_losses, get_folders, write_args
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


class Corpus:
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

        hf, _ = self.fwd_rnn(x)                 # (batch, seq, output_size//2)
        hb, _ = self.bwd_rnn(self._reverse(x))  # (batch, seq, output_size//2)

        # Select final representation.
        hf = hf[:, -1, :]  # (batch, output_size//2)
        hb = hb[:, -1, :]  # (batch, output_size//2)

        h = torch.cat((hf, hb), dim=-1)  # (batch, output_size)
        return self.dropout(self.relu(self.linear(h)))


class TreeDecoder(DiscParser):
    """A tree structured decoder, effectively a DiscRNNG."""
    NUM_ACTIONS = 3
    NUM_WORDS = 1

    def __init__(self, num_words, num_nonterminals, emb_dim, hidden_size,
                 num_layers, latent_dim, dropout, device=None):
        nonterminal_embedding = nn.Embedding(num_nonterminals, emb_dim)
        action_embedding = nn.Embedding(self.NUM_ACTIONS, emb_dim)
        word_embedding = nn.Embedding(self.NUM_WORDS, emb_dim)

        stack_encoder = StackLSTM(
            emb_dim, hidden_size, dropout, device, composition='basic')
        buffer_encoder = BufferLSTM(
            emb_dim, hidden_size, num_layers, dropout, device)
        history_encoder = HistoryLSTM(
            emb_dim, hidden_size, dropout, device)

        super(TreeDecoder, self).__init__(
            word_embedding,
            nonterminal_embedding,
            action_embedding,
            stack_encoder,
            buffer_encoder,
            history_encoder,
            device=device)

        self.device = device
        self.dropout = nn.Dropout(p=dropout)

        self.config2latent = nn.Sequential(
            nn.Linear(3*hidden_size + latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout))
        self.latent2words = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_words))

    def get_input(self):
        stack, buffer, history = self.get_encoded_input()
        return torch.cat((buffer, history, stack), dim=-1)

    def forward(self, sentence, actions, z):
        """Transform z through an RNNG decoder."""
        sentence, actions = deepcopy(sentence), deepcopy(actions)
        # Introduce z via the empty embedding.
        self.stack.empty_emb = nn.Parameter(z)
        self.buffer.empty_emb = nn.Parameter(z)
        self.history.empty_emb = nn.Parameter(z)
        # First encoding with the guards, in this case z.
        self.initialize(sentence)
        leaf_zs = []  # to save the transformations of z at the leaf
        for i, action in enumerate(actions):
            # The state of the parser just before the shift
            # is will be used to compute the transformed z.
            if action == SHIFT:
                u = self.get_input()
                # We compute z_i (we concatenate z for extra representation).
                zi = self.config2latent(torch.cat((u, z), dim=-1))
                leaf_zs.append(zi)
            # Advance the parser with the action.
            self.parse_step(action)
        tree = self.stack.get_tree(with_tag=False)
        leaf_zs = torch.cat(leaf_zs, dim=0)
        x = self.latent2words(leaf_zs)
        return x, tree


class Inference(nn.Module):
    def __init__(self, num_words, emb_dim, hidden_dim, latent_dim, num_layers, dropout, device=None):
        super(Inference, self).__init__()
        self.embedding = nn.Embedding(num_words, emb_dim)
        self.encoder = BiRecurrentEncoder(emb_dim, hidden_dim, hidden_dim, num_layers, dropout, batch_first=True, device=device)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logsigma = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, indices):
        x = self.dropout(self.embedding(indices))
        h = self.encoder(x)
        return self.mu(h), self.logsigma(h)


class TreeVAE(nn.Module):
    def __init__(self, dictionary, emb_dim, hidden_dim, num_layers, latent_dim, dropout, device=None):
        super(TreeVAE, self).__init__()
        self.dictionary = dictionary
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.device = device

        self.generative = TreeDecoder(
            len(dictionary.w2i), len(dictionary.n2i), emb_dim, hidden_dim, num_layers, latent_dim, dropout, device=device)
        self.inference = Inference(
            len(dictionary.w2i), emb_dim, hidden_dim, latent_dim, num_layers, dropout, device=device)

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

    def decode(self, sentence, actions, z):
        return self.generative(sentence, actions, z)

    def forward(self, sentence, actions):
        indices = [self.dictionary.w2i[word.token.processed] for word in sentence]
        x = wrap([indices], self.device)
        mu, logsigma = self.encode(x)
        z = self.sample(mu, logsigma)
        x, tree = self.decode(sentence, actions, z)
        return x, mu, logsigma, tree

    def kl(self, mu, logsigma):
        return -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())


############
#   Main   #
############

def insert_leaves(tree, leaves, dummy_word='+'):
    brackets = tree.split(dummy_word)
    return ''.join(b + l for b, l in zip(brackets[:-1], leaves)) + brackets[-1]


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
    corpus = Corpus(
        data_path=args.data,
        model=args.model,
        textline=args.textline,
        name=args.name,
        use_chars=args.use_chars,
        max_lines=args.max_lines
    )
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

    # Create model.
    model = TreeVAE(corpus.dictionary, args.emb_dim, args.word_lstm_hidden,
        args.lstm_num_layers, args.emb_dim, args.dropout, device=args.device)
    model.to(args.device)

    print(model)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    # criterion = nn.CrossEntropyLoss()

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)

    log = dict(elbo=[], loss=[], kl=[])

    def train():
        # Train.
        for i, (sentence, actions) in enumerate(train_dataset, 1):
            indices = [corpus.dictionary.w2i[word.token.processed] for word in sentence]
            x = wrap([indices], args.device)

            logits, mu, logsigma, tree = model(sentence, actions)
            loss = criterion(logits, x.squeeze(0))
            kl = model.kl(mu, logsigma)
            elbo = loss + kl

            log['elbo'].append(elbo.item())
            log['loss'].append(loss.item())
            log['kl'].append(kl.item())

            optimizer.zero_grad()
            elbo.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            if i % args.print_every == 0:
                elbo = np.mean(log['elbo'][-args.print_every:])
                loss = np.mean(log['loss'][-args.print_every:])
                kl = np.mean(log['kl'][-args.print_every:])
                tensorboard_writer.add_scalar('train/loss', loss, i)
                tensorboard_writer.add_scalar('train/kl', kl, i)
                tensorboard_writer.add_scalar('train/elbo', elbo, i)
                print(f'| Step {i:4d} | ELBO {elbo:.3f} | Loss {loss:.3f} | KL {kl:.3f} | ')

            if i % args.eval_every == 0:
                eval()

    def eval():
        model.eval()
        print('-'*89)
        print('Evaluating...')
        for i, (sentence, actions) in enumerate(dev_dataset[:3]):
            indices = [corpus.dictionary.w2i[word.token.processed] for word in sentence]
            x = wrap([indices], args.device)
            # Compute logits.
            logits, _, _, tree = model(sentence, actions)
            # Decode with argmax.
            predictions = logits.argmax(dim=-1)
            pred = [corpus.dictionary.i2w[i] for i in predictions]
            gold = [corpus.dictionary.i2w[i] for i in indices]
            acc = (np.array(gold) == np.array(pred)).mean()
            print('gold:', ' '.join(gold))
            print('pred:', ' '.join(pred))
            print('acc:', round(acc, 3))
            if args.sample_proposals:
                cat = dist.Categorical(logits.exp())
                print('mean-entropy:', round(cat.entropy().mean().item(), 2))
                for i in range(5):
                    indices = cat.sample()
                    sample = [corpus.dictionary.i2w[k] for k in indices]
                    acc = (indices == x).float().mean().item()
                    print(f'sample {i}:', insert_leaves(tree, sample), f'(acc: {acc:.2f})')
            print()
        print('-'*89)
        model.train()

    try:
        train()
    except KeyboardInterrupt:
        eval()


if __name__ == '__main__':
    main()
