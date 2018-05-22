import torch
import torch.nn as nn
from torch.autograd import Variable

from newdata import PAD_INDEX, EMPTY_INDEX, REDUCED_INDEX, wrap
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
    def __init__(self, vocab_size, stack_size, action_size, emb_dim, emb_dropout,
                lstm_hidden, lstm_num_layers, lstm_dropout, mlp_hidden, cuda):
        super(RNNG, self).__init__()
        self.lstm_hidden = lstm_hidden

        # Embeddings
        self.embedding = nn.Embedding(stack_size, emb_dim, padding_idx=PAD_INDEX)
        # OLD
        self.stack_emb = nn.Embedding(stack_size, emb_dim, padding_idx=PAD_INDEX)
        self.buffer_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_INDEX)
        self.history_emb = nn.Embedding(action_size, emb_dim, padding_idx=PAD_INDEX)
        self.dropout = nn.Dropout(p=emb_dropout)

        # Syntactic composition function called when REDUCE is excecuted
        self.composition = BiRecurrentEncoder(input_size=emb_dim, hidden_size=lstm_hidden,
                                num_layers=lstm_num_layers, batch_first=True,
                                dropout=lstm_dropout, cuda=cuda)

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
        # mlp_input = 3 * 2 * lstm_hidden # three bidirectional lstm embeddings
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

    def forward(self, sent, actions, dictionary, verbose=False):
        """Forward training pass for RNNG.

        Args:
            sent (list): Input sentence as list of indices.
            actions (list): Parse action sequence as list of indices.
            dictionary: an instance of data.Dictionary
        """
        # Create a parser
        parser = Parser(dictionary, self.embedding, self.history_emb)
        parser.initialize(sent)

        # Reinitialize the hidden state of the StackLSTM
        self.stack_lstm.initialize_hidden()

        # Cummulator for loss
        loss = Variable(torch.zeros(1))

        for t, action_idx in enumerate(actions): # t is timestep, i the action index

            action = dictionary.i2a[action_idx] # Get the action as string
            parser.history.push(action_idx)

            if verbose: print('\n{}. '.format(t), parser, action)

            # Comput parse representation and prediction.
            stack, buffer, history = parser.get_embedded_input()
            out = self.encode(stack, buffer, history) # encode the parse configuration
            step_loss = self.loss(out, action_idx)
            loss += step_loss

            if action == 'SHIFT':
                parser.shift()

            elif action == 'REDUCE':
                # Pop all items from the open nonterminal
                tokens, embeddings = parser.stack.pop()
                # Reduce them
                x = self.stack_lstm.reduce(embeddings)
                # Push new representation onto stack
                parser.stack.push(REDUCED_INDEX, vec=x)

                if verbose: print('REDUCEING', [dictionary.i2s[i] for i in tokens])

            elif action.startswith('NT'):
                parser.stack.push(action_idx)

            else:
                raise ValueError('Got unknown action {}'.format(a))

        loss /= len(actions)
        return loss
