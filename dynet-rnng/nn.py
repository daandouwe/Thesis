import dynet as dy


class Affine:
    """Computes affine transformation Wx + b."""
    def __init__(self, model, input_dim, output_dim):
        # self.model = model.add_subcollection("Affine")
        self.weight = model.add_parameters((output_dim, input_dim), init='glorot')
        self.bias = model.add_parameters(output_dim, init='glorot')

    def __call__(self, x):
        return self.weight * x + self.bias


class MLP:
    """A multilayer perceptron with one hidden layer and dropout."""
    def __init__(self, model, input_dim, hidden_dim, output_dim, dropout=0.):
        self.fc1 = Affine(model, input_dim, hidden_dim)
        self.fc2 = Affine(model, hidden_dim, output_dim)
        self.dropout = dropout
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, x):
        h = self.fc1(x)
        h = dy.rectify(h)
        if self.training:
            h = dy.dropout(h, self.dropout)
        return self.fc2(h)
