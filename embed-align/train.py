import csv

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from data import ParallelCorpus
from model import EmbedAlign
from util import Timer

# np.random.seed(42)

########################################
english_path = 'hansards/hansards.36.2.e'
french_path = 'hansards/hansards.36.2.f'

test = True
train = not test

vocab_size = 1000
max_lines = 1000
length_ordered = False
emb_dim = 10
hidden_dim = 10
z_dim = 10
batch_size = 128
num_epochs = 3
learning_rate = 1e-3
print_every = 10
save_every = 10
########################################

corpus = ParallelCorpus(english_path, french_path,
                max_vocab_size=vocab_size, max_lines=max_lines, ordered=length_ordered)
model = EmbedAlign(vocab_size, vocab_size, emb_dim, hidden_dim, z_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

########################################

def save():
    torch.save(model, 'models/model.dict')

def write():
    with open('csv/elbo.csv', 'w') as f:
        names = [["elbo", "-kl", "log_px", "log_py"]]
        logs = list(zip(ELBO, KL, PX, PY))
        writer = csv.writer(f)
        writer.writerows(names + logs)

def plot():
    fig, ax = plt.subplots()
    ax.plot(range(len(ELBO)), ELBO, label='elbo')
    ax.plot(range(len(KL)), KL, label='-kl')
    ax.plot(range(len(PX)), PX, label='log p(x)')
    ax.plot(range(len(PY)), PY, label='log p(y)')
    ax.legend()
    plt.savefig('elbo.pdf')


if test:
    print("Testing")
    batches = corpus.batches(batch_size)
    x, y = next(batches)

    log_px, log_py, kl = model(x, y)

if train:
    ELBO = []
    KL = []
    PX = []
    PY = []
    try:
        timer = Timer()
        for epoch in range(num_epochs):
            batches = corpus.batches(batch_size)
            n_batches = len(corpus.l1.data) // batch_size
            for step, (x, y) in enumerate(batches, 1):
                alpha = 1.

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
                    print("Epoch {} | Step {}/{} | Elbo {:.4f} | {:.0f} sents/sec |\
                            ".format(epoch, step, n_batches, ELBO[-1],
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


########################################
