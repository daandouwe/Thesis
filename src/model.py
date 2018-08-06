import torch
import torch.nn as nn
from torch.autograd import Variable

from data import PAD_INDEX, EMPTY_INDEX, REDUCED_INDEX, REDUCED_TOKEN, wrap
from glove import load_glove
from embedding import ConvolutionalCharEmbedding
from nn import MLP, StackLSTM, HistoryLSTM, BufferLSTM
from parser import Parser

class RNNG(nn.Module):
    """Recurrent Neural Network Grammar model."""
    def __init__(self,
                 dictionary,
                 word_emb_dim,
                 action_emb_dim,
                 word_lstm_hidden,
                 action_lstm_hidden,
                 lstm_num_layers,
                 mlp_hidden,
                 dropout,
                 device=None,
                 use_glove=False,
                 glove_dir=None,
                 glove_error_dir=None,
                 use_char=False):
        super(RNNG, self).__init__()
        self.dictionary = dictionary
        self.device = device

        # Embeddings
        num_words = len(dictionary.w2i)
        num_nonterminals = len(dictionary.n2i)
        num_actions = len(dictionary.a2i)
        if use_char:
            # In this case, num_words is the number of characters.
            self.word_embedding = ConvolutionalCharEmbedding(
                    num_words, word_emb_dim, padding_idx=PAD_INDEX,
                    dropout=dropout, device=device)
        else:
            self.word_embedding = nn.Embedding(num_words, word_emb_dim, padding_idx=PAD_INDEX)
        self.nonterminal_embedding = nn.Embedding(num_nonterminals, word_emb_dim, padding_idx=PAD_INDEX)
        self.action_embedding = nn.Embedding(num_actions, action_emb_dim, padding_idx=PAD_INDEX)

        # Parser encoders
        self.buffer_encoder  = BufferLSTM(word_emb_dim, word_lstm_hidden, lstm_num_layers, dropout, device)
        self.stack_encoder   = StackLSTM(word_emb_dim, word_lstm_hidden, dropout, device)
        self.history_encoder = HistoryLSTM(action_emb_dim, action_lstm_hidden, dropout, device)

        # MLP for action classifiction
        mlp_input = 2 * word_lstm_hidden + action_lstm_hidden
        self.mlp = MLP(mlp_input, mlp_hidden, num_actions, dropout)

        self.dropout = nn.Dropout(p=dropout)

        # Create an internal parser.
        self.parser = Parser(self.dictionary, self.word_embedding,
                             self.nonterminal_embedding, self.action_embedding,
                             self.buffer_encoder, device=device)

        # Training objective
        self.criterion = nn.CrossEntropyLoss()

        if use_glove:
            self.load_glove(word_emb_dim, glove_dir, glove_error_dir)

        self.initialize_parameters()

    def initialize_parameters(self):
        """Initialize parameters with Glorot."""
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)

    def load_glove(self, dim, glovedir, logdir):
        """Load pretrained glove embeddings that are fixed during training."""
        # Make sure to get words by order of index.
        words = [self.dictionary.i2w[i] for i in range(len(self.dictionary.w2i))]
        embeddings = load_glove(words, dim, glovedir, logdir)
        self.word_embedding.weight = nn.Parameter(embeddings)
        self.word_embedding.weight.requires_grad = False

    def encode(self, stack, buffer, history):
        # Apply dropout.
        stack = self.dropout(stack)     # (batch, input_size)
        buffer = self.dropout(buffer)   # (batch, input_size)
        history = self.dropout(history) # (batch, input_size)

        # Encode
        b = buffer # buffer is already the top hidden state
        h = self.history_encoder(history) # returns top hidden state
        s = self.stack_encoder(stack) # returns top hidden state

        # Concatenate and apply mlp to obtain logits.
        x = torch.cat((b, h, s), dim=-1)

        logits = self.mlp(x)
        return logits

    def get_loss(self, logits, y):
        """Compute the loss given the criterion.

        Arguments:
            logits: model predictions.
            y (int): the correct index.
        """
        y = wrap([y], self.device)
        return self.criterion(logits, y)

    def forward(self, sentence, indices, actions, verbose=False, file=None):
        """Forward training pass for RNNG.

        Arguments:
            sentence (list): input sentence as list of words (str).
            indices (list): input sentence as list of indices (int).
            actions (list): parse action sequence as list of indices (int).
        """
        # Initialize the parser with the sentence.
        self.parser.initialize(sentence, indices)

        # Encode the buffer just ones. The hidden representations are
        # stored inside the parser.
        self.parser.buffer.encode()

        # Reset the hidden states of the StackLSTM and the HistoryLSTM.
        self.stack_encoder.initialize_hidden()
        self.history_encoder.initialize_hidden()

        # Cummulator variable for loss
        loss = Variable(torch.zeros(1, device=self.device))

        for t, action_id in enumerate(actions):

            # Less dictionaries
            action = self.dictionary.i2a[action_id] # Get the action as string

            # Compute parse representation and prediction.
            stack, buffer, history = self.parser.get_embedded_input()
            logits = self.encode(stack, buffer, history) # encode the parse configuration
            step_loss = self.get_loss(logits, action_id)
            loss += step_loss

            if verbose:
                # Log parser state.
                print(t, file=file)
                print(str(self.parser), file=file)
                vals, ids = logits.sort(descending=True)
                vals, ids = vals.data.squeeze(0), ids.data.squeeze(0)
                print('Values : ', vals.numpy()[:10], file=file)
                print('Ids : ', ids.numpy()[:10], file=file)
                print('Action : ', action_id, action, file=file)
                print(file=file)

            self.parser.history.push(action_id)

            if action == 'SHIFT':
                self.parser.shift()

            elif action == 'REDUCE':
                # Pop all items from the open nonterminal.
                tokens, indices, embeddings = self.parser.stack.pop()
                # Reduce these items using the composition function.
                x = self.stack_encoder.reduce(embeddings)
                # Push the new representation onto stack: the computed vector x
                # and a dummy index.
                self.parser.stack.push(REDUCED_TOKEN, REDUCED_INDEX, x)

            elif action.startswith('NT'):
                self.parser.stack.open_nonterminal(action, action_id)

            else:
                raise ValueError('got unknown action {}'.format(a))

        return loss


    def parse(self, sentence, indices, verbose=False, file=None):
        """Parse an input sequence.

        Arguments:
            sentence (list): input sentence as list of words (str).
            indices (list): input sentence as list of indices (int).
            actions (list): parse action sequence as list of indices (int).
        """
        # Initialize the parser with the sentence.
        self.parser.initialize(sentence, indices)

        # We encode the buffer just ones. The hidden representations are
        # stored inside the parser.
        self.parser.buffer.encode()

        # Initialize the hidden state of the StackLSTM and the HistoryLSTM.
        self.stack_encoder.initialize_hidden()
        self.history_encoder.initialize_hidden()

        t = 0
        while not self.parser.stack.empty:
            t += 1

            # Compute parse representation and prediction.
            stack, buffer, history = self.parser.get_embedded_input()
            logits = self.encode(stack, buffer, history) # encode the parse configuration

            # Get highest scoring valid predictions.
            vals, ids = logits.sort(descending=True)
            vals, ids = vals.data.squeeze(0), ids.data.squeeze(0)
            i = 0
            action_id = ids[i]
            action = self.dictionary.i2a[action_id]
            while not self.parser.is_valid_action(action):
                i += 1
                action_id = ids[i]
                action = self.dictionary.i2a[action_id]
            self.parser.history.push(action_id)

            # Log info
            if verbose:
                print(t, file=file)
                print(str(self.parser), file=file)
                print('Values : ', vals.numpy()[:10], file=file)
                print('Ids : ', ids.numpy()[:10], file=file)
                print('Action : ', action_id, action, file=file)
                print('Recalls : ', i, file=file)
                print(file=file)

            if action == 'SHIFT':
                self.parser.shift()

            elif action == 'REDUCE':
                # Pop all items from the open nonterminal.
                tokens, indices, embeddings = self.parser.stack.pop()
                # Reduce these items.
                x = self.stack_encoder.reduce(embeddings)
                # Push the new representation onto stack: the computed vector x
                # and a dummy index.
                self.parser.stack.push(REDUCED_TOKEN, REDUCED_INDEX, x)

                if file: print('Reducing : ', tokens, file=file)

            elif action.startswith('NT'):
                self.parser.stack.open_nonterminal(action, action_id)

            else:
                raise ValueError('got illegal action: {}'.format(a))

        return self.parser
