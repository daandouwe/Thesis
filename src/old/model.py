import torch
import torch.nn as nn
from torch.autograd import Variable

from data import PAD_INDEX, EMPTY_INDEX, REDUCED_INDEX, wrap
from nn import MLP, BiRecurrentEncoder, StackLSTM

from parser import Parser

class RNNG(nn.Module):
    """Recurrent Neural Network Grammar model."""
    def __init__(self, dictionary, stack_size, action_size, emb_dim, emb_dropout,
                lstm_hidden, lstm_num_layers, lstm_dropout, mlp_hidden, cuda):
        super(RNNG, self).__init__()
        self.dictionary = dictionary
        self.lstm_hidden = lstm_hidden

        # Embeddings
        self.embedding = nn.Embedding(stack_size, emb_dim, padding_idx=PAD_INDEX)
        self.history_emb = nn.Embedding(action_size, emb_dim, padding_idx=PAD_INDEX)
        self.dropout = nn.Dropout(p=emb_dropout)

        # StackLSTM
        self.stack_lstm = StackLSTM(input_size=emb_dim, hidden_size=lstm_hidden,
                                cuda=cuda)
        # Bidirectional RNN encoders
        self.buffer_encoder = BiRecurrentEncoder(input_size=emb_dim, hidden_size=lstm_hidden,
                                num_layers=lstm_num_layers, batch_first=True,
                                dropout=lstm_dropout, cuda=cuda)
        self.history_encoder = BiRecurrentEncoder(input_size=emb_dim, hidden_size=lstm_hidden,
                                num_layers=lstm_num_layers, batch_first=True,
                                dropout=lstm_dropout, cuda=cuda)

        # MLP for action classifiction
        mlp_input = (2 + 2 + 1) * lstm_hidden # buffer and history are bidirectional, StackLSTM is unidirectional
        self.mlp = MLP(mlp_input, mlp_hidden, action_size)

        # Create an internal parser.
        self.parser = Parser(self.dictionary, self.embedding, self.history_emb)

        # Training objective
        self.criterion = nn.CrossEntropyLoss()

        # To cuda or not to cuda
        self.cuda = cuda

    def encode(self, stack, buffer, history):
        # Apply dropout
        stack = self.dropout(stack) # input_stack (batch, input_size)
        buffer = self.dropout(buffer) # input_stack (batch, input_size)
        history = self.dropout(history) # input_stack (batch, input_size)

        # Encode them
        o = self.buffer_encoder(buffer)
        h = self.history_encoder(history)
        s = self.stack_lstm(stack) # Encode stack. Returns new hidden state.

        # concatenate and apply mlp to obtain logits
        x = torch.cat((o, s, h), dim=-1)
        logits = self.mlp(x)
        return logits

    def loss(self, logits, y):
        """Compute the loss given the criterion.

        Logits is a PyTorch tensor, y is an integer.
        """
        y = wrap([y]) # returns a pytorch Variable
        return self.criterion(logits, y)

    def forward(self, sent, actions, verbose=False, file=None):
        """Forward training pass for RNNG.

        Args:
            sent (list): Input sentence as list of indices.
            actions (list): Parse action sequence as list of indices.
        """
        # Initialize the parser with the sentence.
        self.parser.initialize(sent)

        # Initialize the hidden state of the StackLSTM.
        self.stack_lstm.initialize_hidden()

        # Cummulator for loss
        loss = Variable(torch.zeros(1))

        for t, action_id in enumerate(actions):

            # Less dictionaries
            action = self.dictionary.i2a[action_id] # Get the action as string

            # Compute parse representation and prediction.
            stack, buffer, history = self.parser.get_embedded_input()
            out = self.encode(stack, buffer, history) # encode the parse configuration
            step_loss = self.loss(out, action_id)
            loss += step_loss

            if verbose:
                # Log parser state.
                print(t, file=file)
                print(str(self.parser), file=file)
                vals, ids = out.sort(descending=True)
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
                tokens, embeddings = self.parser.stack.pop()
                # Reduce these items using the composition function.
                x = self.stack_lstm.reduce(embeddings)
                # Push new representation onto stack.
                self.parser.stack.push(REDUCED_INDEX, vec=x)

            elif action.startswith('NT'):
                self.parser.stack.push(action_id, new_nonterminal=True)

            else:
                raise ValueError('Got unknown action {}'.format(a))

        loss /= len(actions) # Average loss over the action sequence

        return loss


    def parse(self, sent, file):
        """Parse an input sequence.

        Args:
            sent (list): input sentence as list of indices
        """
        # Initialize the parser with the sentence.
        self.parser.initialize(sent)

        # Initialize the hidden state of the StackLSTM.
        self.stack_lstm.initialize_hidden()

        # Cummulator for loss.
        loss = Variable(torch.zeros(1))

        t = 0
        while not self.parser.stack.empty:
            t += 1

            # Compute parse representation and prediction.
            stack, buffer, history = self.parser.get_embedded_input()
            out = self.encode(stack, buffer, history) # encode the parse configuration

            # Get highest scoring valid predictions.
            vals, ids = out.sort(descending=True)
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
                tokens, embeddings = self.parser.stack.pop()
                # Reduce them
                x = self.stack_lstm.reduce(embeddings)
                # Push new representation onto stack:
                # the computed vector x and a dummy index.
                self.parser.stack.push(REDUCED_INDEX, vec=x)

                print('Reducing : ', [self.dictionary.i2s[i] for i in tokens], file=file)

            elif action.startswith('NT'):
                self.parser.stack.push(action_id, new_nonterminal=True)

            else:
                raise ValueError('got illegal action: {}'.format(a))

        return self.parser
