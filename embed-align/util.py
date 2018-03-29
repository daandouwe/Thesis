import time
import torch
from aer import test

class Timer:
    """A simple timer to track speed of training"""
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

def align(x, y, py):
    """
    Computes posterior alignment for one batch.
    """
    y = y.unsqueeze(1).expand(-1, x.size(1), -1) # [batch_size, l1_sent_len, l2_sent_len]
    x_mask = (x > 0).float() # [batch_size, l1_sent_len] (0 is padding index)
    x_mask = x_mask.unsqueeze(-1).expand(-1, -1, y.size(-1)) # [batch_size, l1_sent_len, l2_vocab_size]
    selected = torch.gather(py, -1, y) * x_mask
    _, a = selected.max(dim=1)
    return a.data.numpy()

def eval_alignments(gold_path, pred_path):
    """
    A useless wrapper but the name is better.
    """
    aer = test(gold_path, pred_path)
    return aer

def predict_alignments(model, batches, write_path='predicted/dev.align'):
    with open(write_path, 'w') as f:
        for k, (x, y) in enumerate(batches):
            batch_size, sent_len = y.shape
            mu, sigma = model.encoder(x)
            z = model.sample(mu, sigma)
            py = model.g(z)           # [batch_size, l1_sent_len, l2_vocab_size]
            a = align(x, y, py)
            for i in range(batch_size):
                batch_num = k*batch_size
                for j in range(sent_len):
                    if y[i][j].data.numpy() == 0: # Start of padding
                        break
                    print(batch_num + i+1, a[i][j]+1, j+1, 'S', file=f)

if __name__ == '__main__':
    # Testing the annealing
    anneal = AnnealKL(rate=10)
    for i in range(100):
        print(i, anneal.alpha(update=i))
