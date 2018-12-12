import time
import functools
import math

import dynet as dy
import numpy as np

import trees
import vocabulary
import parse

def main():

    treebank = trees.load_trees(
        '/Users/daan/data/ptb-benepar/02-21.10way.clean', strip_top=True)

    # Convert each tree to CNF
    treebank = [tree.convert().binarize() for tree in treebank[:1000]]

    # Obtain the word an label vocabularies
    words = [leaf.word for tree in treebank for leaf in tree.leaves()]
    labels = [label for tree in treebank for label in tree.labels()]

    word_vocab = vocabulary.Vocabulary()
    word_vocab.index(parse.START)
    word_vocab.index(parse.STOP)
    word_vocab.index(parse.UNK)
    for word in words:
        word_vocab.index(word)

    label_vocab = vocabulary.Vocabulary()
    label_vocab.index((trees.DUMMY,))
    for label in labels:
        label_vocab.index(label)

    print(label_vocab.values)

    model = dy.ParameterCollection()
    parser = parse.ChartParser(
        model,
        word_vocab,
        label_vocab,
        word_embedding_dim=10,
        lstm_layers=1,
        lstm_dim=10,
        span_hidden_dim=10,
        label_hidden_dim=10,
        dropout=0.,
    )
    optimizer = dy.AdamTrainer(model)

    total_loss = 0

    for i, tree in enumerate(treebank, 1):
        dy.renew_cg()

        words = [leaf.word for leaf in tree.leaves()]

        t0 = time.time()
        loss, _, _ = parser.parse(words, gold=tree)
        t1 = time.time()

        loss.forward()
        loss.backward()
        optimizer.update()

        t2 = time.time()

        total_loss += loss.value()

        print('step', i, 'loss', round(total_loss/i, 2), 'forward-time', round(t1-t0, 3), 'backward-time', round(t2-t1, 3), 'length', len(words))


if __name__ == '__main__':
    main()
