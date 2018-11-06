import dynet as dy


class Affine:
    """Computes affine transformation Wx + b."""
    def __init__(self, model, input_dim, output_dim):

        self.model = model.add_subcollection("Affine")

        self.weight = self.model.add_parameters((output_dim, input_dim), init='glorot')
        self.bias = self.model.add_parameters(output_dim, init='glorot')

    def __call__(self, x):
        return self.weight * x + self.bias


class Feedforward:
    """Feedforward network with relu nonlinearity and dropout."""
    def __init__(self, model, input_dim, hidden_dims, output_dim, dropout=0.):

        self.model = model.add_subcollection('Feedforward')

        self.layers =  []
        dims = [input_dim] + hidden_dims + [output_dim]
        for prev_dim, next_dim in zip(dims, dims[1:]):
            self.layers.append(Affine(self.model, prev_dim, next_dim))
        self.dropout = dropout
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = dy.rectify(x)
            if self.training:
                x = dy.dropout(x, self.dropout)
        return x
