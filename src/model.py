import torch
import torch.nn as nn
from torch.autograd import Variable

from data import PAD_INDEX

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class RNNG(nn.Module):
    def __init__(self,
                 vocab_size,
                 stack_size,
                 action_size,
                 emb_dim,
                 emb_dropout,
                 lstm_hidden,
                 lstm_num_layers,
                 lstm_dropout,
                 mlp_hidden):
        super(RNNG, self).__init__()

        # Embeddings
        self.stack_emb = nn.Embedding(stack_size, emb_dim, padding_idx=PAD_INDEX)
        self.buffer_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_INDEX)
        self.history_emb = nn.Embedding(action_size, emb_dim, padding_idx=PAD_INDEX)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        # LSTMs
        lstm_input = emb_dim
        self.stack_lstm = nn.LSTM(input_size=lstm_input, hidden_size=lstm_hidden,
                                num_layers=lstm_num_layers, batch_first=True,
                                dropout=lstm_dropout)
        self.buffer_lstm = nn.LSTM(input_size=lstm_input, hidden_size=lstm_hidden,
                                num_layers=lstm_num_layers, batch_first=True,
                                dropout=lstm_dropout)
        self.history_lstm = nn.LSTM(input_size=lstm_input, hidden_size=lstm_hidden,
                                num_layers=lstm_num_layers, batch_first=True,
                                dropout=lstm_dropout)

        # MLP for classifiction
        self.mlp = MLP(lstm_hidden, mlp_hidden, action_size)

    def forward(self, stack, buffer, history):
        stack = self.emb_dropout(self.stack_emb(stack))
        buffer = self.emb_dropout(self.buffer_emb(buffer))
        history = self.emb_dropout(self.history_emb(history))

        stack_forward = self.stack_lstm(stack)
        buffer_forward = self.buffer_lstm(buffer)
        history_forward = self.history_lstm(history)



        x = torch.cat((stack, buffer, history), dim=-1)
        out =  self.MLP(x)
        return out
