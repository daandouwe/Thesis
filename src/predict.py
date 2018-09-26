#!/usr/bin/env python
import argparse
import os
import glob

import torch

from data import Corpus, get_sentences


def predict(model, batches, pred_path):
    model.eval()
    nsents = len(batches)
    trees = []
    with torch.no_grad():  # operations inside don't track history.
        for i, batch in enumerate(batches):
            sentence, actions = batch
            tree = model.parse(sentence)
            # Modify the gold data with the list of actions.
            trees.append(tree)
            if i % 10 == 0:
                print(f'Predicting sentence {i}/{nsents}...', end='\r')
    with open(pred_path, 'w') as f:
        print('\n'.join(trees), file=f)


def main(args):
    checkpath = os.path.join(args.checkdir, 'model.pt')
    print(f'Loading model from {checkpath}...')
    model = torch.load(checkpath)
    print(f'Loading data from {args.data}...')
    corpus = Corpus(data_path=args.data, textline=args.textline)
    test_batches  = corpus.test.batches(length_ordered=False, shuffle=False)
    predict(args, model, test_batches, name='test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose')
    parser.add_argument('-l', '--use_latest', action='store_true',
                        help='use the latest predictions')
    parser.add_argument('--data', type=str, default='../tmp',
                        help='location of the data corpus')
    parser.add_argument('--textline', type=str, choices=['unked', 'lower', 'upper'], default='unked',
                        help='textline to use from the oracle file')
    parser.add_argument('--folder', type=str, default='',
                        help='the folder in outdir to look for')
    parser.add_argument('--name', type=str, choices=['train', 'dev', 'test'], default='test',
                        help='name of the data set used')
    args = parser.parse_args()

    if args.use_latest:
        # Folder names start with a timestamp sorting gives latest.
        args.checkdir = max(glob.glob(os.path.join('checkpoints', '*/')))
        args.outdir = max(glob.glob(os.path.join('out', '*/')))
    elif args.folder:
        args.checkdir = os.path.join('checkpoints', args.folder)
        args.outdir = os.path.join('out', args.folder)
    else:
        exit('Either pass -l (--use_latest) or --folder must be specified.')
    main(args)
