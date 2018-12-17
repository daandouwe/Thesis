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
    treebank = [tree.cnf() for tree in treebank[:10000]]

    # Obtain the word an label vocabularies
    words = [word for tree in treebank for word in tree.words()]
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
        word_embedding_dim=100,
        lstm_layers=2,
        lstm_dim=256,
        span_hidden_dim=256,
        label_hidden_dim=256,
        dropout=0.5,
    )
    optimizer = dy.AdamTrainer(model)

    total_loss = 0

    test_tree = treebank[3]

    for i, tree in enumerate(treebank, 1):
        dy.renew_cg()

        t0 = time.time()
        loss = parser.forward(tree)
        t1 = time.time()

        loss.forward()
        loss.backward()
        optimizer.update()

        t2 = time.time()

        total_loss += loss.value()

        print('step', i, 'loss', round(total_loss/i, 2), 'forward-time', round(t1-t0, 3), 'backward-time', round(t2-t1, 3))

        if i % 50 == 0:
            pred, _ = parser.parse(test_tree.words())
            print('='*50)
            print('>', test_tree.un_cnf().linearize())
            print('>', pred.un_cnf().linearize())
            samples = parser.sample(test_tree.words(), num_samples=20)
            for tree, _ in samples:
                print('>', tree.un_cnf().linearize())
            print('='*50)


if __name__ == '__main__':
    main()
