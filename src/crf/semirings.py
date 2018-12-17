import dynet as dy


class Semiring(object):
    pass


class ProbSemiring(Semiring):

    @staticmethod
    def sum(x, y):
        return x + y

    @staticmethod
    def product(x, y):
        return x * y

    @staticmethod
    def division(x, y):
        return dy.cdiv(x, y)

    @staticmethod
    def as_real(x):
        return x

    @staticmethod
    def sums(xs):
        """Compute the sum over all values."""
        return dy.esum(xs)

    @staticmethod
    def products(values):
        """Compute the product over all values."""
        k = dy.ones(1)
        for value in values:
            k = k * value
        return k

    @classmethod
    def zero(cls):
        return dy.zeros(1)

    @classmethod
    def one(cls):
        return dy.ones(1)


class LogProbSemiring(Semiring):

    @staticmethod
    def sum(x, y):
        return dy.logsumexp([x, y])

    @staticmethod
    def product(x, y):
        return x + y

    @staticmethod
    def division(x, y):
        return x - y

    @staticmethod
    def as_real(x):
        return dy.exp(x)

    @staticmethod
    def sums(xs):
        """Compute the sum over all values."""
        return dy.logsumexp(xs)

    @staticmethod
    def products(xs):
        """Compute the product over all values."""
        return dy.esum(xs)

    @staticmethod
    def zero():
        return dy.constant(1, -np.inf)

    @staticmethod
    def one():
        return dy.zeros(1)


class ViterbiSemiring(Semiring):

    @staticmethod
    def sum(x, y):
        return dy.emax([x, y])

    @staticmethod
    def product(x, y):
        return x + y

    @staticmethod
    def division(x, y):
        return x - y

    @staticmethod
    def as_real(x):
        return dy.exp(x)

    @staticmethod
    def sums(xs):
        """Compute the sum over all values."""
        return dy.emax(xs)

    @staticmethod
    def products(xs):
        """Compute the product over all values."""
        return dy.esum(xs)

    @staticmethod
    def zero():
        return dy.constant(1, -np.inf)

    @staticmethod
    def one():
        return dy.zeros(1)
