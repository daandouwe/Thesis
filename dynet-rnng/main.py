#!/usr/bin/env python
from typing import NamedTuple
import time

import dynet as dy
import numpy as np

from data import Corpus
from model import DiscRNNG
from actions import SHIFT, REDUCE, NT, is_nt, get_nt


BATCH_SIZE = 32
PRINT_EVERY = 10
EVAL_EVERY = 100


def clock_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    hours, minutes, seconds = int(hours), int(minutes), int(seconds)
    return "{}h{:02}m{:02}s".format(hours, minutes, seconds)


def main():

    corpus = Corpus(
        train_path='../data/train/ptb.train.oracle',
        dev_path='../data/dev/ptb.dev.oracle',
        test_path='../data/test/ptb.test.oracle'
    )

    model = dy.Model()
    trainer = dy.AdamTrainer(model)

    rnng = DiscRNNG(
        model=model,
        dictionary=corpus.dictionary,
        num_words=len(corpus.dictionary.w2i),
        num_nt=len(corpus.dictionary.n2i),
        word_emb_dim=100,
        nt_emb_dim=100,
        action_emb_dim=20,
        stack_hidden_size=100,
        buffer_hidden_size=100,
        history_hidden_size=20,
        stack_num_layers=2,
        buffer_num_layers=2,
        history_num_layers=2,
        mlp_hidden=100,
        dropout=0.3,
        device=None
    )

    train_batches = corpus.train.batches(BATCH_SIZE)
    test_sentence = train_batches[0][0][0]
    num_batches = len(train_batches)
    t0 = time.time()
    losses = []
    for step, minibatch in enumerate(train_batches, 1):
        dy.renew_cg()

        loss = dy.esum([rnng(words, actions) for words, actions in minibatch])
        loss /= BATCH_SIZE

        loss.forward()
        loss.backward()
        trainer.update()

        losses.append(loss.value())
        avg_loss = np.mean(losses[-PRINT_EVERY:])
        elapsed = time.time() - t0
        updates_per_sec = step / elapsed
        sents_per_sec = BATCH_SIZE * updates_per_sec
        eta = (num_batches - step) / updates_per_sec


        if step % PRINT_EVERY == 0:
            print(f'| {step}/{num_batches} ({step/num_batches:.0%}) | Loss {avg_loss:.3f} | Elapsed {clock_time(elapsed)} | Eta {clock_time(eta)} | {sents_per_sec:.1f} sents/sec | {updates_per_sec:.1f} updates/sec |')

        if step % EVAL_EVERY == 0:
            dy.renew_cg()
            tree, nll = rnng.parse(test_sentence)
            print(tree.linearize(with_tag=False), -nll.value())


if __name__ == '__main__':
    main()
