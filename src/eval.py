#!/usr/bin/env python

import os
import glob
import argparse

from PYEVALB.scorer import Scorer

from get_vocab import get_sentences

def oracle2tree(sent):
    """Returns a linearize tree from a list of actions in an oracle file.

    Arguments:
        sent (dictionary): format returned by get_sent_dict from get_vocab.py
    """
    actions   = sent['actions']
    words     = sent['upper'].split()
    tags      = sent['tags'].split()
    gold_tree = sent['tree'][2:] # remove the hash at the beginning
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
    oracle_path = os.path.join(args.preddir, args.name + '.pred.oracle')
    pred_path   = os.path.join(args.preddir, args.name + '.pred.trees')
    gold_path   = os.path.join(args.preddir, args.name + '.gold.trees')
    result_path = os.path.join(args.preddir, args.name + '.result')

    predicted_sents = get_sentences(oracle_path)
    with open(pred_path, 'w') as f:
        with open(gold_path, 'w') as g:
            for sent in predicted_sents:
                pred_tree, gold_tree = oracle2tree(sent)
                print(pred_tree, file=f)
                print(gold_tree, file=g)
    scorer = Scorer()
    scorer.evalb(gold_path, pred_path, result_path)

    if args.verbose:
        with open(result_path) as f:
            print(f.read())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose')
    parser.add_argument('-l', '--use_latest', action='store_true',
                        help='use the latest predictions')
    parser.add_argument('--outdir', type=str, default='out',
                        help='directory where predictions are written to')
    parser.add_argument('--folder', type=str, default='',
                        help='the folder in outdir to look for')
    parser.add_argument('--name', type=str, choices=['train', 'dev', 'test'], default='test',
                        help='name of the data set used')
    args = parser.parse_args()

    if args.use_latest:
        # folder names start with a timestamp
        latest_dir = max(glob.glob(os.path.join(args.outdir, '*/')))
        args.preddir = latest_dir
    else:
        assert args.folder, 'if not using latest a folder must be specified.'
        args.preddir = os.path.join(args.outdir, args.folder)

    main(args)
