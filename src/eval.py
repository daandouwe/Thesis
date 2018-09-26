#!/usr/bin/env python
import os
import glob
import argparse
import re
import subprocess

from data import get_sentences


def evalb(evalb_dir, pred_path, gold_path, result_path, ignore_error=10000):
    """Use original EVALB to score trees."""
    evalb_dir = os.path.expanduser(evalb_dir)
    assert os.path.exists(evalb_dir), f'Do you have EVALB installed at {evalb_dir}?'
    evalb_exec = os.path.join(evalb_dir, "evalb")
    command = '{} {} {} -e {} > {}'.format(
        evalb_exec,
        pred_path,
        gold_path,
        ignore_error,
        result_path
    )
    subprocess.run(command, shell=True)
    # Read result path and get F-sore.
    with open(result_path) as f:
        for line in f:
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore = float(match.group(1))
                return fscore
    return 0.0


def actions2tree(words, actions, tags=None):
    """Returns a linearizen tree from a list of words and actions."""
    # reverse words:
    words = words[::-1]
    if tags:
        tags = tags[::-1]
    tree = ''
    for i, a in enumerate(actions):
        if a == 'SHIFT':
            word = words.pop()
            if tags:
                tag = tags.pop()
                tree += f'({tag} {word}) '
            else:
                tree += f'{word} '
        elif a == 'REDUCE':
            tree = tree[:-1] # Remove whitespace
            tree += ') '
        else:
            tree += f'({a} '
    return tree


def eval_oracle(outdir, verbose=False):
    oracle_path = os.path.join(outdir, 'test.pred.oracle')
    pred_path   = os.path.join(outdir, 'test.pred.trees')
    gold_path   = os.path.join(outdir, 'test.gold.trees')
    result_path = os.path.join(outdir, 'test.result')
    predicted_sents = get_sentences(oracle_path)
    with open(pred_path, 'w') as f:
        with open(gold_path, 'w') as g:
            for sent in predicted_sents:
                pred_tree = actions2tree(sent['upper'].split(),
                                         sent['actions'],
                                         sent['tags'].split())
                gold_tree = sent['tree'][2:] # remove the hash at the beginning
                print(pred_tree, file=f)
                print(gold_tree, file=g)
    scorer = Scorer()
    scorer.evalb(gold_path, pred_path, result_path)
    if verbose:
        with open(result_path) as f:
            print(f.read())


def main(args):
    oracle_path = os.path.join(args.preddir, args.name + '.pred.oracle')
    pred_path   = os.path.join(args.preddir, args.name + '.pred.trees')
    gold_path   = os.path.join(args.preddir, args.name + '.gold.trees')
    result_path = os.path.join(args.preddir, args.name + '.result')
    predicted_sents = get_sentences(oracle_path)
    with open(pred_path, 'w') as f:
        with open(gold_path, 'w') as g:
            for sent in predicted_sents:
                pred_tree = actions2tree(sent['upper'].split(),
                                         sent['actions'],
                                         sent['tags'].split())
                gold_tree = sent['tree'][2:] # remove the hash at the beginning
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
    parser.add_argument('--name', type=str, choices=['train', 'dev', 'test'],
                        default='test', help='name of the data set used')
    args = parser.parse_args()

    if args.use_latest:
        # folder names start with a timestamp
        latest_dir = max(glob.glob(os.path.join(args.outdir, '*/')))
        args.preddir = latest_dir
    else:
        assert args.folder, 'if not using latest a folder must be specified.'
        args.preddir = os.path.join(args.outdir, args.folder)

    main(args)
