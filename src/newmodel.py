from copy import deepcopy

import torch
import torch.nn as nn
from torch.autograd import Variable

from data import PAD_INDEX, EMPTY_INDEX
from stacklstm import StackLSTM

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

        self.cuda = cuda

    def wrap(self, x):
        if self.cuda:
            return Variable(torch.cuda.LongTensor(x))
        else:
            return Variable(torch.LongTensor(x))

    def forward(self, sent, actions):
        """
        sent = [2, 4, 6, 2]
        actions = [2, 4, 6, 2]
        """
        # Start with full buffer, and empyt history and empty stack
        buffer = [w for w in sent[::-1]] # buffer is sentence reversed
        history = [EMPTY_INDEX]
        stack = [EMPTY_INDEX]

        # Package the lists
        input_stack = self.wrap(stack) # (batch, seq_len)
        input_buffer = self.wrap([buffer]) # the entire buffer (batch, 1, seq_len)
        input_history = self.wrap([history]) # the entire stack (batch, 1, seq_len)
        # Embed them
        input_stack = self.stack_emb(input_stack) # input_stack (batch, input_size)
        input_buffer = self.buffer_emb(input_buffer) # input_stack (batch, input_size)
        input_history = self.history_emb(input_history) # input_stack (batch, input_size)
        # Apply dropout
        input_stack = self.dropout(input_stack) # input_stack (batch, input_size)
        input_buffer = self.dropout(input_buffer) # input_stack (batch, input_size)
        input_history = self.dropout(input_history) # input_stack (batch, input_size)
        # Encode them
        o = self.buffer_encoder(input_buffer)
        h = self.history_encoder(input_history)
        self.stack_lstm.initialize_hidden()
        s = self.stack_lstm(input_stack) # Encode stack. Returns latest new state.
        print(o.shape, s.shape, h.shape)

        x = torch.cat((o, s, h), dim=-1)
        out = self.mlp(x)
        print(out.shape)
        exit()

        history.pop() # Remove EMPTY_INDEX from list
        stack.pop() # Remove EMPTY_INDEX from list
        for i, a in enumerate(actions):
            # a is an integer
            if i2a[a] == 'SHIFT':
                stack.append(buffer.pop()) # move top word from buffer to stack
                self.stack_lstm.stack
                print(stack)

            elif a == REDUCE:
                # This is the hardest
                raise NotImplementedError
                # self.reduce()

            else:
                raise NotImplementedError

            # Create representation of configuration
            x = torch.cat((o, s, h), dim=-1)
            out = mlp(x)
