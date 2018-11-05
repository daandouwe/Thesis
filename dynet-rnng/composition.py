import dynet as dy

from nn import Affine


class BiRecurrentComposition:
    """Bidirectional RNN composition function."""
    def __init__(self, model, input_size, num_layers, dropout):
        assert input_size % 2 == 0, 'input size size must be even'
        self.fwd_rnn_builder = dy.VanillaLSTMBuilder(num_layers, input_size, input_size//2, model)
        self.bwd_rnn_builder = dy.VanillaLSTMBuilder(num_layers, input_size, input_size//2, model)
        self.fwd_rnn_builder.set_dropouts(dropout, dropout)
        self.bwd_rnn_builder.set_dropouts(dropout, dropout)

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
    """Bidirectional RNN composition function with attention."""
    def __init__(self, model, input_size, num_layers, dropout):
        assert input_size % 2 == 0, 'hidden size must be even'
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.fwd_rnn_builder = dy.VanillaLSTMBuilder(num_layers, input_size, input_size//2, model)
        self.bwd_rnn_builder = dy.VanillaLSTMBuilder(num_layers, input_size, input_size//2, model)
        self.V = model.add_parameters((input_size, input_size), init='glorot')
        self.gating = Affine(model, 2*input_size, input_size)
        self.head = Affine(model, input_size, input_size)
        self.training = True

    def __call__(self, head, children):
        if self.training:
            self.fwd_rnn_builder.set_dropouts(self.dropout, self.dropout)
            self.bwd_rnn_builder.set_dropouts(self.dropout, self.dropout)
        else:
            self.fwd_rnn_builder.disable_dropout()
            self.bwd_rnn_builder.disable_dropout()
        fwd_rnn = self.fwd_rnn_builder.initial_state()
        bwd_rnn = self.bwd_rnn_builder.initial_state()
        hf = fwd_rnn.transduce([head] + children)  # ['NP', 'the', 'cat']
        hb = bwd_rnn.transduce([head] + children[::-1])  # ['NP', 'cat', 'the']
        hf = dy.concatenate(hf, d=1)  # (input_size//2, seq_len)
        hb = dy.concatenate(hb, d=1)  # (input_size//2, seq_len)
        h = dy.concatenate([hf, hb], d=0)  # (input_size, seq_len)

        a = dy.transpose(h) * self.V * head  # (seq_len,)
        a = dy.softmax(a, d=0)  # (seq_len,)

        m = h * a  # (input_size,)

        x = dy.concatenate([head, m], d=0)  # (2*input_size,)
        g = dy.logistic(self.gating(x))  # (input_size,)
        t = self.head(head)  # (input_size,)
        c = dy.cmult(g, t) + dy.cmult((1 - g), m)  # (input_size,)

        if not self.training:
            # Store internally for inspection during prediction.
            self._attn = a.value()
            self._gate = g.value()
        return c
