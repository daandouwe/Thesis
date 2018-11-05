#!/usr/bin/env python
import argparse
import os

ACTIONS = ('SHIFT', 'REDUCE', 'OPEN')

# TODO: load this from utils
# But why does not work `from ..utils import get_sentences`?

def get_sentences(path):
    """Chunks the oracle file into sentences."""
    def get_sent_dict(sent):
        d = {
                'tree'     : sent[0],
                'tags'     : sent[1],
                'original' : sent[2],
                'lower'    : sent[3],
                'unked'    : sent[4],
                'actions'  : sent[5:]
            }
        return d

    sentences = []
    with open(path) as f:
        sent = []
        for line in f:
            if line == '\n':
                sentences.append(sent)
                sent = []
            else:
                sent.append(line.rstrip())
        return [get_sent_dict(sent) for sent in sentences if sent]


def get_vocab(sentences, textline='unked'):
    """Returns the vocabulary used in the oracle file."""
    textline_options = sentences[0].keys()
    assert textline in textline_options, 'invalid choice of textline: choose from {}'.format(list(textline_options))
    vocab = set()
    for sent_dict in sentences:
        vocab.update(set(sent_dict[textline].split()))
    vocab = sorted(list(vocab))
    return vocab


def get_nonterminals(sentences):
    """Returns the set of actions used in the oracle file."""
    nonterminals = set()
    for sent_dict in sentences:
        nts = [a[3:-1] for a in sent_dict['actions'] if a.startswith('NT')]
        nonterminals.update(nts)
    nonterminals = sorted(list(nonterminals))
    return nonterminals


def get_actions(sentences):
    """Returns the set of actions used in the oracle file."""
    return ACTIONS


def main(args):
    # Partition the oracle files into sentences
    train = get_sentences(args.train)
    dev = get_sentences(args.dev)
    test = get_sentences(args.test)
    sentences = train + dev + test

    # Collect desired symbols for our dictionaries
    actions = get_actions(sentences)
    vocab = get_vocab(sentences, textline=args.textline)  # TODO: oov?
    nonterminals = get_nonterminals(sentences)

    # Write out vocabularies
    path = os.path.join(args.outdir, args.name)
    print('\n'.join(nonterminals),
            file=open(path + '.nonterminals', 'w'))
    print('\n'.join(actions),
            file=open(path + '.actions', 'w'))
    print('\n'.join(vocab),
            file=open(path + '.vocab', 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data for RNNG parser.')
    parser.add_argument('train', type=str,
                        help='path to train oracle.')
    parser.add_argument('dev', type=str,
                        help='path to dev oracle.')
    parser.add_argument('test', type=str,
                        help='path to test oracle.')
    parser.add_argument('--name', type=str, default='ptb',
                        help='name of dataset')
    parser.add_argument('--outdir', type=str, default='../tmp',
                        help='path to output vocabulary')
    parser.add_argument('--textline', type=str, choices=['unked', 'lower', 'original'], default='unked',
                        help='textline to use from the oracle file')
    args = parser.parse_args()

    main(args)
