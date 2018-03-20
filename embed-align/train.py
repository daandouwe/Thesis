import csv

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from data import ParallelCorpus
from model import EmbedAlign
from util import Timer, AnnealKL, predict_alignments, eval_alignments

np.random.seed(42)

########################################
e_train_path = 'hansards/train/train.e'
e_dev_path = 'hansards/dev/dev.e'
e_test_path = 'hansards/test/test.e'
f_train_path = 'hansards/train/train.f'
f_dev_path = 'hansards/dev/dev.f'
f_test_path = 'hansards/test/test.f'

test = False
train = not test

l1_vocab_size = 10000
l2_vocab_size = 10000
max_lines = None
length_ordered = False
emb_dim = 50
hidden_dim = 50
z_dim = 50
batch_size = 32
num_epochs = 5
learning_rate = 1e-3
print_every = 10
save_every = 1000
write_every = 100
mean_sent = True

########################################
corpus = ParallelCorpus(e_train_path, e_dev_path, e_test_path,
                          f_train_path, f_dev_path, f_test_path,
                          l1_vocab_size, l2_vocab_size,
                          max_lines, ordered=length_ordered)
model = EmbedAlign(l1_vocab_size, l2_vocab_size, emb_dim, hidden_dim, z_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

########################################
def save(path='models/model.dict'):
    torch.save(model, path)

def write(path='csv/elbo.csv'):
    with open(path, 'w') as f:
        writer = csv.writer(f)
        names = [["elbo", "-kl", "log_px", "log_py"]]
        logs = list(zip(ELBO, KL, PX, PY))
        writer.writerows(names + logs)
    with open('csv/aer.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows([['aer']] + [[aer] for aer in AER])

def evaluate():
    predict_alignments(model, corpus.dev_batches(batch_size))
    aer = eval_alignments('hansards/dev/dev.wa.nonullalign', 'predicted/dev.align')
    return aer

if test:
    print("Testing")
    batches = corpus.batches(batch_size)
    x, y = next(batches)
    log_px, log_py, kl = model(x, y, mean_sent=mean_sent)

    aer = evaluate()
    print(aer)

if train:
    ELBO = []
    KL = []
    PX = []
    PY = []
    AER = []
    try:
        timer = Timer()
        anneal = AnnealKL(step=1e-3, rate=10)
        for epoch in range(num_epochs):
            batches = corpus.batches(batch_size)
            n_batches = len(corpus.l1.train.data) // batch_size
            for step, (x, y) in enumerate(batches, 1):
                # alpha = anneal.alpha(epoch*n_batches + step)
                alpha = 0.1

                log_px, log_py, kl = model(x, y)
                elbo = log_px + log_py - alpha*kl
                loss = -elbo

                # Update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Store for plotting
                ELBO.append(elbo.data.numpy()[0])
                KL.append(-kl.data.numpy()[0])
                PX.append(log_px.data.numpy()[0])
                PY.append(log_py.data.numpy()[0])

                if step % print_every == 0:
                    aer = evaluate()
                    AER.append(aer)
                    print('Epoch {} | Step {}/{} | Elbo {:.4f} | AER {:.2f} | alpha {:.3f} | {:.0f} sents/sec |'
                            ''.format(epoch, step, n_batches, np.mean(ELBO[-print_every:]),
                                np.mean(AER[-print_every:]), alpha, batch_size*print_every/timer.elapsed()), end='\r')

                if step % save_every == 0:
                    save()

                if step % write_every == 0:
                    write()

        save()
        write()

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    save()
    write()
