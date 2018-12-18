import time

import dynet as dy

import utils.trees as trees
import utils.vocabulary as vocabulary
from crf.semirings import LogProbSemiring, ProbSemiring
from crf.feedforward import Feedforward
from crf.model import ChartParser, START, STOP


def main():

    with open('/Users/daan/data/ptb-benepar/23.auto.clean') as f:
        lines = [line.strip() for line in f.readlines()]

    treebank = [trees.fromstring(line, strip_top=True) for line in lines[:100]]
    tree = treebank[0].cnf()

    # Obtain the word an label vocabularies
    words = [vocabulary.UNK, START, STOP] + [word for word in tree.words()]
    labels = [(trees.DUMMY,)] + [label for tree in treebank[:100] for label in tree.cnf().labels()]

    word_vocab = vocabulary.Vocabulary.fromlist(words, unk_value=vocabulary.UNK)
    label_vocab = vocabulary.Vocabulary.fromlist(labels)

    model = dy.ParameterCollection()
    parser = ChartParser(
        model,
        word_vocab,
        label_vocab,
        word_embedding_dim=100,
        lstm_layers=2,
        lstm_dim=100,
        label_hidden_dim=100,
        dropout=0.,
    )
    optimizer = dy.AdamTrainer(model)

    for i in range(1000):
        dy.renew_cg()

        t0 = time.time()
        loss = parser.forward(tree)
        pred, _ = parser.parse(tree.words())
        sample, _ = parser.sample(tree.words())
        t1 = time.time()

        loss.forward()
        loss.backward()
        optimizer.update()

        t2 = time.time()

        print('step', i, 'loss', round(loss.value(), 2), 'forward-time', round(t1-t0, 3), 'backward-time', round(t2-t1, 3), 'length', len(words))
        print('>', tree.un_cnf().linearize(with_tag=False))
        print('>', pred.linearize(with_tag=False))
        print('>', sample.linearize(with_tag=False))
        print()

if __name__ == '__main__':
    main()
