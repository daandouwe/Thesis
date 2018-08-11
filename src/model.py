import os

import torch
import torch.nn as nn

from data import PAD_INDEX, REDUCED_INDEX, REDUCED_TOKEN, Item, Action
from glove import load_glove, get_vectors
from embedding import ConvolutionalCharEmbedding
from nn import MLP
from encoder import StackLSTM, HistoryLSTM, BufferLSTM
from parser import Parser
from loss import LossCompute


class RNNG(Parser):
    """Recurrent Neural Network Grammar model."""
    def __init__(
        self,
        dictionary,
        actions,
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
        device
    ):
        super(RNNG, self).__init__(
            word_embedding,
            nonterminal_embedding,
            action_embedding,
            buffer_encoder,
            actions,
            device=device
        )
        self.dictionary = dictionary
        self.device = device

        self.word_embedding = word_embedding
        self.nonterminal_embedding = nonterminal_embedding
        self.action_embedding = action_embedding

        # Parser encoders
        self.buffer_encoder = buffer_encoder
        self.stack_encoder = stack_encoder
        self.history_encoder = history_encoder

        # MLP for action classifiction
        self.action_mlp = action_mlp
        # MLP for nonterminal classifiction
        self.nonterminal_mlp = nonterminal_mlp

        self.dropout = nn.Dropout(p=dropout)

        # Loss computation
        self.loss_compute = loss_compute

    def encode(self, stack, buffer, history):
        # Apply dropout.
        stack = self.dropout(stack)      # (batch, input_size)
        buffer = self.dropout(buffer)    # (batch, input_size)
        history = self.dropout(history)  # (batch, input_size)
        # Encode
        b = buffer  # buffer is already the top hidden state
        h = self.history_encoder(history)  # returns top hidden state
        s = self.stack_encoder(stack)  # returns top hidden state
        # Concatenate and apply mlp to obtain logits.
        x = torch.cat((b, h, s), dim=-1)
        return x

    def parse_step(self, action, logfile=False):
        """Updates parser one step give the action."""
        self.history.push(action)
        if action.index == self.SHIFT:
            self.shift()
        elif action.index == self.REDUCE:
            # Pop all items from the open nonterminal.
            items, embeddings = self.stack.pop()
            # Reduce these items using the composition function.
            x = self.stack_encoder.reduce(embeddings)
            # Push the new representation onto stack: the computed vector x
            # and a dummy index.
            self.stack.push(
                Item(REDUCED_TOKEN, REDUCED_INDEX, embedding=x),
                reduced=True
            )
        elif action.index == self.OPEN:
            item = Item(
                action.symbol.token, action.symbol.index, nonterminal=True
            )
            self.stack.open_nonterminal(item)
        else:
            raise ValueError(f'got illegal action: {action.token}')

    def forward(self, sentence, actions, logfile=None):
        """Forward training pass for RNNG.

        Arguments:
            sentence (list): input sentence as list of Item objects.
            actions (list): parse action sequence as list of Item objects.
        """
        # Initialize the parser with the sentence.
        self.initialize(sentence)
        # Reset the hidden states of the StackLSTM and the HistoryLSTM.
        self.stack_encoder.initialize_hidden()
        self.history_encoder.initialize_hidden()
        # Cummulator variable for loss
        loss = torch.zeros(1, device=self.device)
        for t, action in enumerate(actions):
            # Get the parser representation.
            stack, buffer, history = self.get_embedded_input()
            # Encode it.
            x = self.encode(stack, buffer, history)
            # Compute the logits for the action.
            action_logits = self.action_mlp(x)
            loss += self.loss_compute(action_logits, action.index)
            # If we open a nonterminal, predict which.
            if action.index == self.OPEN:
                nonterminal_logits = self.nonterminal_mlp(x)
                loss += self.loss_compute(nonterminal_logits, action.symbol.index)
            # Take the appropriate parse step.
            self.parse_step(action)
        return loss

    def parse(self, sentence, logfile=None):
        """Parse an input sequence.

        Arguments:
            sentence (list): input sentence as list of Item objects.
        """
        # Initialize the parser with the sentence.
        self.initialize(sentence)
        # Reset the hidden states of the StackLSTM and the HistoryLSTM.
        self.stack_encoder.initialize_hidden()
        self.history_encoder.initialize_hidden()
        t = 0
        while not self.stack.empty:
            t += 1
            # Get the parser representation.
            stack, buffer, history = self.get_embedded_input()
            # Encode it.
            x = self.encode(stack, buffer, history)
            # Compute the logits for the action.
            action_logits = self.action_mlp(x)
            # Get highest scoring valid predictions.
            vals, ids = action_logits.sort(descending=True)
            vals, ids = vals.data.squeeze(0), ids.data.squeeze(0)
            i = 0
            action = Action(self.dictionary.i2a[ids[i]], ids[i])
            while not self.is_valid_action(action):
                i += 1
                action = Action(self.dictionary.i2a[ids[i]], ids[i])
            if action.index == self.OPEN:
                nonterminal_logits = self.nonterminal_mlp(x)
                vals, ids = nonterminal_logits.sort(descending=True)
                vals, ids = vals.data.squeeze(0), ids.data.squeeze(0)
                action.symbol = Item(self.dictionary.i2n[ids[0]], ids[0], nonterminal=True)
            # Take the appropriate parse step.
            self.parse_step(action)
        return self.actions


def set_embedding(embedding, tensor):
    assert tensor.shape == embedding.weight.shape
    embedding.weight = nn.Parameter(tensor)
    embedding.weight.requires_grad = False


def make_model(args, dictionary):
    # Embeddings
    num_words = dictionary.num_words
    num_nonterminals = dictionary.num_nonterminals
    num_actions = dictionary.num_actions
    a2i = dictionary.a2i
    actions = (a2i['SHIFT'], a2i['REDUCE'], a2i['OPEN'])
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
    action_embedding = nn.Embedding(num_actions, args.action_emb_dim, padding_idx=PAD_INDEX)

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
        args.action_emb_dim,
        args.action_lstm_hidden,
        args.dropout,
        args.device
    )

    mlp_input = 2 * args.word_lstm_hidden + args.action_lstm_hidden
    action_mlp = MLP(
        mlp_input,
        args.mlp_dim,
        num_actions,
        args.dropout
    )
    nonterminal_mlp = MLP(
        mlp_input,
        args.mlp_dim,
        num_nonterminals,
        args.dropout
    )

    loss_compute = LossCompute(nn.CrossEntropyLoss, args.device)

    model = RNNG(
        dictionary=dictionary,
        actions=actions,
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

    # Initialize parameters with Glorot.
    for param in model.parameters():
        if param.dim() > 1 and param.requires_grad:
            nn.init.xavier_uniform_(param)

    return model
