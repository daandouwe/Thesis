import dynet as dy


class StackLSTM:
    """An LSTM with a pop operation."""
    def __init__(self, model, input_size, hidden_size, num_layers, dropout):
        assert (hidden_size % 2 == 0), f'hidden size must be even: {hidden_size}'

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_builder = dy.VanillaLSTMBuilder(num_layers, input_size, hidden_size, model)

    def train(self):
        self.rnn_builder.set_dropouts(self.dropout, self.dropout)

    def eval(self):
        self.rnn_builder.disable_dropout()

    def initialize(self):
        self.rnn = self.rnn_builder.initial_state()

    def __call__(self, x):
        # Update the RNN with the input.
        self.rnn = self.rnn.add_input(x)
        # Return the new output.
        return self.rnn.output()

    def push(self, *args, **kwargs):
        return self(*args, **kwargs)

    def pop(self):
        self.rnn = self.rnn.prev()
        return self.rnn.output()

    @property
    def top(self):
        return self.rnn.output()
