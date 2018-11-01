import dynet as dy


class Affine:
    """Computes affine transformation Wx + b."""
    def __init__(self, model, input_size, output_size):
        self.W = model.add_parameters((output_size, input_size), init='glorot')
        self.b = model.add_parameters(output_size, init='glorot')

    def __call__(self, x):
        return self.W * x + self.b


class MLP:
    """A simple multilayer perceptron with one hidden layer and dropout."""
    def __init__(self, model, input_size, hidden_size, output_size, dropout=0.):
        self.fc1 = Affine(model, input_size, hidden_size)
        self.fc2 = Affine(model, hidden_size, output_size)
        self.dropout = dropout

    def __call__(self, x):
        h = self.fc1(x)
        h = dy.rectify(h)
        h = dy.dropout(h, self.dropout)
        return self.fc2(h)
