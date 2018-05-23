import torch
import torch.nn as nn
from torch.autograd import Variable

from data import PAD_INDEX, EMPTY_INDEX, REDUCED_INDEX, wrap
from stacklstm import StackLSTM
from parser import Parser

class MLP(nn.Module):
    """A simple multilayer perceptron with one hidden layer and dropout."""
    def __init__(self, input_size, hidden_size, output_size, dropout=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class BiRecurrentEncoder(nn.Module):
    """A bidirectional RNN encoder."""
    def __init__(self,input_size, hidden_size, num_layers, dropout, batch_first=True, cuda=False):
        super(BiRecurrentEncoder, self).__init__()
        self.forward_rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=batch_first,
                           dropout=dropout)
        self.backward_rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=batch_first,
                           dropout=dropout)
        self.cuda = cuda

    def _reverse(self, tensor):
        idx = [i for i in range(tensor.size(1) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx))
        idx = idx.cuda() if self.cuda else idx
        return tensor.index_select(1, idx)

    def forward(self, x):
        hf, _ = self.forward_rnn(x)                 # [batch, seq, hidden_size]
        hb, _ = self.backward_rnn(self._reverse(x)) # [batch, seq, hidden_size]

        # select final representation
        hf = hf[:, -1, :] # [batch, hidden_size]
        hb = hb[:, -1, :] # [batch, hidden_size]

        h = torch.cat((hf, hb), dim=-1) # [batch, 2*hidden_size]
        return h


class RNNG(nn.Module):
    def __init__(self, stack_size, action_size, emb_dim, emb_dropout,
                lstm_hidden, lstm_num_layers, lstm_dropout, mlp_hidden, cuda):
        super(RNNG, self).__init__()
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
        mlp_input = (2*2 + 1) * lstm_hidden # two bidirectional lstm embeddings, one unidirectional
        self.mlp = MLP(mlp_input, mlp_hidden, action_size)

        self.criterion = nn.CrossEntropyLoss()

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

        # concatenate and apply mlp to obtain output
        x = torch.cat((o, s, h), dim=-1)
        out = self.mlp(x)
        return out

    def loss(self, logits, y):
        y = wrap([y])
        return self.criterion(logits, y)

    def forward(self, sent, actions, dictionary, verbose=False, file=None):
        """Forward training pass for RNNG.

        Args:
            sent (list): Input sentence as list of indices.
            actions (list): Parse action sequence as list of indices.
            dictionary: an instance of data.Dictionary
        """
        # Create a new parser
        parser = Parser(dictionary, self.embedding, self.history_emb)
        parser.initialize(sent)

        # Reinitialize the hidden state of the StackLSTM
        self.stack_lstm.initialize_hidden()

        # Cummulator for loss
        loss = Variable(torch.zeros(1))

        for t, action_id in enumerate(actions):

            action = dictionary.i2a[action_id] # Get the action as string
            parser.history.push(action_id)

            # Comput parse representation and prediction.
            stack, buffer, history = parser.get_embedded_input()
            out = self.encode(stack, buffer, history) # encode the parse configuration
            step_loss = self.loss(out, action_id)
            loss += step_loss

            if verbose:
                # Log parser state
                print(t, file=file)
                print(str(parser), file=file)
                vals, ids = out.sort(descending=True)
                vals, ids = vals.data.squeeze(0), ids.data.squeeze(0)
                print('Values : ', vals.numpy()[:10], file=file)
                print('Ids : ', ids.numpy()[:10], file=file)
                print('Action : ', action_id, action, file=file)
                print(file=file)

            if action == 'SHIFT':
                parser.shift()

            elif action == 'REDUCE':
                # Pop all items from the open nonterminal
                tokens, embeddings = parser.stack.pop()
                # Reduce them
                x = self.stack_lstm.reduce(embeddings)
                # Push new representation onto stack
                parser.stack.push(REDUCED_INDEX, vec=x)

            elif action.startswith('NT'):
                parser.stack.push(action_id, new_nonterminal=True)

            else:
                raise ValueError('Got unknown action {}'.format(a))

        loss /= len(actions)
        return loss


    def parse(self, sent, dictionary, file):
        """Forward training pass for RNNG.

        Args:
            sent (list): input sentence as list of indices
            dictionary: an instance of data.Dictionary
        """
        # Create a new parser
        parser = Parser(dictionary, self.embedding, self.history_emb)
        parser.initialize(sent)

        # Reinitialize the hidden state of the StackLSTM
        self.stack_lstm.initialize_hidden()

        # Cummulator for loss
        loss = Variable(torch.zeros(1))

        t = 0
        while not parser.stack.empty:
            t += 1


            # Comput parse representation and prediction.
            stack, buffer, history = parser.get_embedded_input()
            out = self.encode(stack, buffer, history) # encode the parse configuration

            # Get highest scoring valid predictions
            vals, ids = out.sort(descending=True)
            vals, ids = vals.data.squeeze(0), ids.data.squeeze(0)
            i = 0
            action_id = ids[i]
            action = dictionary.i2a[action_id]
            while not parser.is_valid_action(action):
                i += 1
                action_id = ids[i]
                action = dictionary.i2a[action_id]
            parser.history.push(action_id)

            # Log info
            print(t, file=file)
            print(str(parser), file=file)
            print('Values : ', vals.numpy()[:10], file=file)
            print('Ids : ', ids.numpy()[:10], file=file)
            print('Action : ', action_id, action, file=file)
            print('Recalls : ', i, file=file)
            print(file=file)

            if action == 'SHIFT':
                parser.shift()

            elif action == 'REDUCE':
                # Pop all items from the open nonterminal
                tokens, embeddings = parser.stack.pop()
                # Reduce them
                x = self.stack_lstm.reduce(embeddings)
                # Push new representation onto stack
                parser.stack.push(REDUCED_INDEX, vec=x)

                print('Reducing : ', [dictionary.i2s[i] for i in tokens], file=file)

            elif action.startswith('NT'):
                parser.stack.push(action_id, new_nonterminal=True)

            else:
                raise ValueError('got unknown action: {}'.format(a))

        return parser
