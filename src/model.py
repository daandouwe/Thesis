import torch
import torch.nn as nn
from torch.autograd import Variable

from data import PAD_INDEX, EMPTY_INDEX, REDUCED_INDEX, REDUCED_TOKEN, wrap, load_glove
from nn import MLP, BiRecurrentEncoder, StackLSTM, HistoryLSTM
from parser import Parser

class RNNG(nn.Module):
    """Recurrent Neural Network Grammar model."""
    def __init__(self, dictionary, emb_dim, emb_dropout,
                lstm_hidden, lstm_num_layers, lstm_dropout, mlp_hidden,
                cuda, use_glove=False):
        super(RNNG, self).__init__()
        self.dictionary = dictionary
        self.lstm_hidden = lstm_hidden

        ## Embeddings
        vocab_size = len(dictionary.w2i)
        nt_size = len(dictionary.n2i)
        action_size = len(dictionary.a2i)
        # For words...
        self.word_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_INDEX)
        # nonterminals...
        self.nt_embedding = nn.Embedding(nt_size, emb_dim, padding_idx=PAD_INDEX)
        # and actions.
        self.action_embedding = nn.Embedding(action_size, emb_dim, padding_idx=PAD_INDEX)

        self.dropout = nn.Dropout(p=emb_dropout)

        # Parser encoders
        self.stack_lstm = StackLSTM(input_size=emb_dim, hidden_size=lstm_hidden,
                                cuda=cuda)
        self.history_lstm = HistoryLSTM(input_size=emb_dim, hidden_size=lstm_hidden,
                                    cuda=cuda)
        self.buffer_encoder = nn.LSTM(input_size=emb_dim, hidden_size=lstm_hidden,
                                    batch_first=True, dropout=lstm_dropout)

        # MLP for action classifiction
        mlp_input = 3 * lstm_hidden
        self.mlp = MLP(mlp_input, mlp_hidden, action_size)

        # Create an internal parser.
        self.parser = Parser(self.dictionary, self.word_embedding,
                             self.nt_embedding, self.action_embedding)

        # Training objective
        self.criterion = nn.CrossEntropyLoss()

        # To cuda or not to cuda
        self.cuda = cuda

        if use_glove:
            self.load_glove()

    def load_glove(self, path='~/glove'):
        """Load pretrained glove embeddings that are fixed during training.

        TODO: Far too much of our vocabulary does not have pretrained embeddings."""
        embeddings = load_glove(self.dictionary)
        self.word_embedding.weight = nn.Parameter(embeddings)
        self.word_embedding.weight.requires_grad = False

    def encode(self, stack, buffer, history):
        # Apply dropout
        # stack = self.dropout(stack) # input_stack (batch, input_size)
        # buffer = self.dropout(buffer) # input_stack (batch, input_size)
        # history = self.dropout(history) # input_stack (batch, input_size)

        # Encode
        b = buffer # buffer is already the lstm hidden state.
        h = self.history_lstm(history) # Returns top hidden state.
        s = self.stack_lstm(stack) # Returns top hidden state.

        # concatenate and apply mlp to obtain logits
        x = torch.cat((b, h, s), dim=-1)
        logits = self.mlp(x)
        return logits

    def loss(self, logits, y):
        """Compute the loss given the criterion.

        Logits is a PyTorch tensor, y is an integer.
        """
        y = wrap([y]) # returns a pytorch Variable
        return self.criterion(logits, y)

    def forward(self, sentence, indices, actions, verbose=False, file=None):
        """Forward training pass for RNNG.

        Args:
            sentence (list): Input sentence as list of words.
            indices (list): Input sentence as list of indices.
            actions (list): Parse action sequence as list of indices.
        """
        # Initialize the parser with the sentence.
        self.parser.initialize(sentence, indices)

        # Initialize the hidden state of the StackLSTM and the HistoryLSTM.
        self.stack_lstm.initialize_hidden()
        self.history_lstm.initialize_hidden()

        # We encode the buffer just ones, at the beginning, using the lstm
        # buffer_encoder. The resulting representations are stored inside the parser.
        self.parser.buffer.encode(self.buffer_encoder)

        # Cummulator variable for loss
        loss = Variable(torch.zeros(1))

        for t, action_id in enumerate(actions):

            # Less dictionaries
            action = self.dictionary.i2a[action_id] # Get the action as string

            # Compute parse representation and prediction.
            stack, buffer, history = self.parser.get_embedded_input()
            logits = self.encode(stack, buffer, history) # encode the parse configuration
            step_loss = self.loss(logits, action_id)
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
                x = self.stack_lstm.reduce(embeddings)
                # Push new representation onto stack.
                self.parser.stack.push(REDUCED_TOKEN, REDUCED_INDEX, x)

            elif action.startswith('NT'):
                self.parser.stack.open_nonterminal(action, action_id)

            else:
                raise ValueError('Got unknown action {}'.format(a))

        loss /= len(actions) # Average loss over the action sequence

        return loss


    def parse(self, sentence, indices, file):
        """Parse an input sequence.

        Args:
            sent (list): input sentence as list of indices
        """
        # Initialize the parser with the sentence.
        self.parser.initialize(sentence, indices)

        # Initialize the hidden state of the StackLSTM and the HistoryLSTM.
        self.stack_lstm.initialize_hidden()
        self.history_lstm.initialize_hidden()

        # We encode the buffer just ones. The hidden representations are
        # stored in the parser.
        self.parser.buffer.encode(self.buffer_encoder)

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
                # Reduce them
                x = self.stack_lstm.reduce(embeddings)
                # Push new representation onto stack:
                # the computed vector x and a dummy index.
                self.parser.stack.push(REDUCED_TOKEN, REDUCED_INDEX, x)

                print('Reducing : ', tokens, file=file)

            elif action.startswith('NT'):
                self.parser.stack.open_nonterminal(action, action_id)

            else:
                raise ValueError('got illegal action: {}'.format(a))

        return self.parser
