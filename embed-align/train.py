import csv

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from data import ParallelCorpus
from model import EmbedAlign
from util import Timer, AnnealKL

# np.random.seed(42)

########################################
english_path = 'hansards/hansards.36.2.e'
french_path = 'hansards/hansards.36.2.f'

test = False
train = not test

vocab_size = 100
max_lines = None
length_ordered = False
emb_dim = 50
hidden_dim = 100
z_dim = 2
batch_size = 128
num_epochs = 5
learning_rate = 1e-2
print_every = 10
save_every = 1000
write_every = 100
mean_sent = True

########################################
corpus = ParallelCorpus(english_path, french_path,
                max_vocab_size=vocab_size, max_lines=max_lines, ordered=length_ordered)
model = EmbedAlign(vocab_size, vocab_size, emb_dim, hidden_dim, z_dim)
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

def plot(path='elbo.pdf'):
    fig, ax = plt.subplots()
    ax.plot(range(len(ELBO)), ELBO, label='elbo')
    ax.plot(range(len(KL)), KL, label='-kl')
    ax.plot(range(len(PX)), PX, label='log p(x)')
    ax.plot(range(len(PY)), PY, label='log p(y)')
    ax.legend()
    plt.savefig(path)


if test:
    print("Testing")
    batches = corpus.batches(batch_size)
    x, y = next(batches)

    log_px, log_py, kl = model(x, y, mean_sent=mean_sent)

if train:
    ELBO = []
    KL = []
    PX = []
    PY = []
    try:
        timer = Timer()
        anneal = AnnealKL(step=1e-3, rate=10)
        for epoch in range(num_epochs):
            batches = corpus.batches(batch_size)
            n_batches = len(corpus.l1.data) // batch_size
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
                    print("Epoch {} | Step {}/{} | Elbo {:.4f} | alpha {:.3f} | {:.0f} sents/sec |\
                            ".format(epoch, step, n_batches, np.mean(ELBO[-print_every:]), alpha,
                                batch_size*print_every/timer.elapsed()), end='\r')

                if step % save_every == 0:
                    save()

                if step % write_every == 0:
                    write()

        save()
        write()
        plot()

    except KeyboardInterrupt:
        save()
        write()
        plot()
