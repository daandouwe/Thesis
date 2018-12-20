import dynet as dy

from components.feedforward import Affine


class BiRecurrentComposition:
    """Bidirectional RNN composition function."""
    def __init__(self, model, input_size, num_layers, dropout):
        assert input_size % 2 == 0, 'input size size must be even'

        self.model = model.add_subcollection('BiRecurrentComposition')

        self.fwd_rnn_builder = dy.VanillaLSTMBuilder(num_layers, input_size, input_size//2, self.model)
        self.bwd_rnn_builder = dy.VanillaLSTMBuilder(num_layers, input_size, input_size//2, self.model)
        self.dropout = dropout

    def train(self):
        self.fwd_rnn_builder.set_dropouts(dropout, dropout)
        self.bwd_rnn_builder.set_dropouts(dropout, dropout)

    def eval(self):
        self.fwd_rnn_builder.disable_dropout()
        self.bwd_rnn_builder.disable_dropout()

    def __call__(self, head, children):
        fwd_rnn = self.fwd_rnn_builder.initial_state()
        bwd_rnn = self.bwd_rnn_builder.initial_state()
        for x in [head] + children:  # ['NP', 'the', 'hungry', 'cat']
            fwd_rnn = fwd_rnn.add_input(x)
        for x in [head] + children[::-1]:  # ['NP', 'cat', 'hungry', 'the']
            bwd_rnn = bwd_rnn.add_input(x)
        hf = fwd_rnn.output()
        hb = bwd_rnn.output()
        return dy.concatenate([hf, hb], d=0)


class AttentionComposition:
    """Bidirectional RNN composition function with gated attention."""
    def __init__(self, model, input_size, num_layers, dropout):
        assert input_size % 2 == 0, 'hidden size must be even'

        self.model = model.add_subcollection('AttentionComposition')

        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn = dy.BiRNNBuilder(
            num_layers,
            input_size,
            input_size,
            self.model,
            dy.VanillaLSTMBuilder)
        self.V = self.model.add_parameters((input_size, input_size), init='glorot')
        self.gating = Affine(self.model, 2*input_size, input_size)
        self.head = Affine(self.model, input_size, input_size)
        self.training = True

    def train(self):
        self.rnn.set_dropout(self.dropout)
        self.training = True

    def eval(self):
        self.rnn.disable_dropout()
        self.training = False

    def __call__(self, head, children):
        h = self.rnn.transduce(children)
        h = dy.concatenate(h, d=1)  # (input_size, seq_len)

        a = dy.transpose(h) * self.V * head  # (seq_len,)
        a = dy.softmax(a, d=0)  # (seq_len,)
        # a = dy.sparsemax(a)  # (seq_len,)

        m = h * a  # (input_size,)

        x = dy.concatenate([head, m], d=0)  # (2*input_size,)
        g = dy.logistic(self.gating(x))  # (input_size,)
        t = self.head(head)  # (input_size,)
        c = dy.cmult(g, t) + dy.cmult((1 - g), m)  # (input_size,)

        # Store internally for inspection during prediction.
        if not self.training:
            self._attn = a.value()
            self._gate = g.value()
        return c
