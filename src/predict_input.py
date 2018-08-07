#!/usr/bin/env python
import glob
import os

import torch

from data import Corpus
from model import RNNG
from eval import actions2tree
from scripts.get_oracle import unkify

def process_unks(sent, dictionary):
    new_sent = []
    for word in sent:
        try:
            dictionary[word]
            new_sent.append(word)
        except KeyError:
            unk = unkify([word], dictionary)[0]
            new_sent.append(unk)
    return new_sent

def predict(model, sent, dictionary):
    model.eval()
    indices = [dictionary[w] for w in sent]
    parser = model.parse(sent, indices)
    return parser.actions

def main(args):
    # Set cuda.
    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    print('Device: {}.'.format(args.device))

    print('Loading corpus with textline {}.'.format(args.textline))
    corpus = Corpus(data_path=args.data, textline=args.textline, char=args.use_char)
    dictionary = corpus.dictionary.w2i

    latest_dir = max(glob.glob(os.path.join('checkpoints', '*/')))
    checkfile = os.path.join(latest_dir, 'model.pt')
    print('Loading model from {}.'.format(checkfile))
    # Load best saved model.
    with open(checkfile, 'rb') as f:
        model = torch.load(f)
    model.to(args.device)

    while True:
        sent = input('Input a sentence: ')
        words = sent.split()
        unked = process_unks(words, dictionary)
        actions = predict(model, unked, dictionary)
        tree = actions2tree(words, actions[1:]) # Hack...
        print(words)
        print(unked)
        print(tree)
        print()

if __name__ == '__main__':
    main(args)
