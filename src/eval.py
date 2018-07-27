#!/usr/bin/env python

import os
import argparse

from PYEVALB.scorer import Scorer

from get_vocab import get_sentences

def oracle2tree(sent):
    """Returns a linearize tree from a list of actions in an oracle file.

    Arguments:
        sent: a dictionary as returned by get_sent_dict in get_configs.py
    """
    actions = sent['actions']
    words = sent['upper'].split()
    tags = sent['tags'].split()
    gold_tree = sent['tree'][2:] # remove the hash
    pred_tree = ''
    # reverse words:
    words = words[::-1]
    tags = tags[::-1]
    for i, a in enumerate(actions):
        if a == 'SHIFT':
            w = words.pop()
            t = tags.pop()
            pred_tree += '({} {}) '.format(t, w)
        elif a == 'REDUCE':
            pred_tree += ') '
        else:
            nt = a[3:-1] # a is NT(X), and we select only X
            pred_tree += '({} '.format(nt)
    return pred_tree, gold_tree

def main(args):
    path = os.path.join(args.outdir, args.data)
    oracle_path = path + '.pred.oracle'
    pred_path   = path + '.pred.trees'
    gold_path   = path + '.gold.trees'
    result_path = path + '.result'

    predicted_sents = get_sentences(oracle_path)
    with open(pred_path, 'w') as f:
        with open(gold_path, 'w') as g:
            for sent in predicted_sents:
                pred_tree, gold_tree = oracle2tree(sent)
                print(pred_tree, file=f)
                print(gold_tree, file=g)
    scorer = Scorer()
    scorer.evalb(gold_path, pred_path, result_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('outdir', type=str,
                        help='directory where predictions are written to')
    parser.add_argument('--data', type=str, choices=['train', 'dev', 'test'], default='test',
                        help='directory where predictions are written to')
    args = parser.parse_args()

    main(args)
