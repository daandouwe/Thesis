#!/usr/bin/env python
import argparse
import os
import glob

import torch

from data import Corpus
from scripts.get_vocab import get_sentences

def predict(args, model, batches, name='test'):
    model.eval()
    sentences = get_sentences(os.path.join(args.data, 'test', 'ptb.{}.oracle'.format(name)))
    nsents = len(batches)
    for i, batch in enumerate(batches):
        sentence, actions = batch
        parser = model.parse(sent, indices)
        sentences[i]['actions'] = parser.actions
        if i % 10 == 0:
            print('Predicting: sentence {}/{}.'.format(i, nsents), end='\r')
    print()
    write_prediction(sentences, args.outdir, name=name)

def print_sent_dict_as_config(sent_dict, file):
    print(sent_dict['tree'], file=file)
    print(sent_dict['tags'], file=file)
    print(sent_dict['upper'], file=file)
    print(sent_dict['lower'], file=file)
    print(sent_dict['unked'], file=file)
    print('\n'.join(sent_dict['actions']), file=file)
    print(file=file)

def write_prediction(sentences, outdir, name, verbose=False):
    path = os.path.join(outdir, '{}.pred.oracle'.format(name))
    with open(path, 'w') as f:
        for i, sent_dict in enumerate(sentences):
            if verbose: print(i, end='\r')
            print_sent_dict_as_config(sent_dict, f)

def main(args):
    checkpath = os.path.join(args.checkdir, 'model.pt')
    model = torch.load(checkpath)
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
