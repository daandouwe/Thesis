import torch
import torch.nn as nn
from torch.autograd import Variable

from data import PAD_INDEX

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

    def _select_final(self, tensor, lens):
        lens = lens.unsqueeze(1).unsqueeze(2)       # [batch, 1, 1]
        lens = lens.expand(-1, -1, tensor.size(2))  # [batch, 1, sent_len]
        return torch.gather(tensor, 1, lens).squeeze(1)

    def forward(self, xf, xb, lens):
        hf, _ = self.forward_rnn(xf)
        hb, _ = self.backward_rnn(xb)

        hf = self._select_final(hf, lens)
        hb = self._select_final(hb, lens)

        h = torch.cat((hf, hb), dim=-1)
        return h

    def _forward(self, x, lens):
        hf, _ = self.forward_rnn(x)
        hb, _ = self.backward_rnn(self._reverse(x))

        hf = self._select_final(hf, lens)
        hb = self._select_final(hb, lens) # TODO: flip the selection indices

        h = torch.cat((hf, hb), dim=-1)
        return h

class RNNG(nn.Module):
    def __init__(self, vocab_size, stack_size, action_size, emb_dim, emb_dropout,
                lstm_hidden, lstm_num_layers, lstm_dropout, mlp_hidden, cuda):
        super(RNNG, self).__init__()

        # Embeddings
        self.stack_emb = nn.Embedding(stack_size, emb_dim, padding_idx=PAD_INDEX)
        self.buffer_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_INDEX)
        self.history_emb = nn.Embedding(action_size, emb_dim, padding_idx=PAD_INDEX)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        # RNN encoders
        lstm_input = emb_dim
        self.stack_encoder = BiRecurrentEncoder(input_size=lstm_input, hidden_size=lstm_hidden,
                                num_layers=lstm_num_layers, batch_first=True,
                                dropout=lstm_dropout, cuda=cuda)
        self.buffer_encoder = BiRecurrentEncoder(input_size=lstm_input, hidden_size=lstm_hidden,
                                num_layers=lstm_num_layers, batch_first=True,
                                dropout=lstm_dropout, cuda=cuda)
        self.history_encoder = BiRecurrentEncoder(input_size=lstm_input, hidden_size=lstm_hidden,
                                num_layers=lstm_num_layers, batch_first=True,
                                dropout=lstm_dropout, cuda=cuda)

        # MLP for action classifiction
        mlp_input = 3 * 2 * lstm_hidden # three bidirectional lstm embeddings
        self.mlp = MLP(mlp_input, mlp_hidden, action_size)

    def _get_lengths(self, x, dim=1):
        return (x != PAD_INDEX).long().sum(dim) - 1

    def forward(self, stack, buffer, history):
        """stack, buffer, and history are tuples (forward order, backward order)"""
        s_lens = self._get_lengths(stack[0])
        b_lens = self._get_lengths(buffer[0])
        h_lens = self._get_lengths(history[0])

        # Embed each sequence
        s_fw = self.emb_dropout(self.stack_emb(stack[0]))
        b_fw = self.emb_dropout(self.buffer_emb(buffer[0]))
        h_fw = self.emb_dropout(self.history_emb(history[0]))

        s_bw = self.emb_dropout(self.stack_emb(stack[1]))
        b_bw = self.emb_dropout(self.buffer_emb(buffer[1]))
        h_bw = self.emb_dropout(self.history_emb(history[1]))

        # Bidirectional RNN encoding
        hs = self.stack_encoder(s_fw, s_bw, s_lens)
        hb = self.buffer_encoder(b_fw, b_bw, b_lens)
        hh = self.history_encoder(h_fw, h_bw, h_lens)

        # Create representation of configuration
        x = torch.cat((hs, hb, hh), dim=-1)
        out =  self.mlp(x)
        return out

    def _forward(self, stack, buffer, history):
        s_lens = self._get_lengths(stack)
        b_lens = self._get_lengths(buffer)
        h_lens = self._get_lengths(history)

        # Embed each sequence
        s = self.emb_dropout(self.stack_emb(stack))
        b = self.emb_dropout(self.buffer_emb(buffer))
        h = self.emb_dropout(self.history_emb(history))

        # Bidirectional RNN encoding
        hs = self.stack_encoder(s, s_lens)
        hb = self.buffer_encoder(b, b_lens)
        hh = self.history_encoder(h, h_lens)

        # Create representation of configuration
        x = torch.cat((hs, hb, hh), dim=-1)
        out =  self.mlp(x)
        return out
