import time

class Timer:
    def __init__(self):
        self.time0 = time.time()

    def elapsed(self):
        time1 = time.time()
        elapsed = time1 - self.time0
        self.time0 = time1
        return elapsed


class AnnealKL:
    def __init__(self, step=1e-3, rate=500):
        self.rate = rate
        self.step = step

    def alpha(self, update):
        n, _ = divmod(update, self.rate)
        return max(1., n*self.step)


if __name__ == '__main__':
    # Testing the annealing
    anneal = AnnealKL(rate=10)
    for i in range(100):
        print(i, anneal.alpha(update=i))
