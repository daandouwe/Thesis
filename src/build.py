"""
Build a vocabulary.
"""

import argparse
import json
from collections import Counter

from utils.trees import fromstring
from utils.text import BRACKETS, replace_brackets, replace_quotes


def main(args):
    print('Building vocabulary.')

    print(f'Loading labeled data from `{args.train_path}`...')
    with open(args.train_path) as f:
        ptb_data = [fromstring(line.strip()) for line in f]

    ptb_words = [word for tree in ptb_data for word in tree.words()]

    if args.unlabeled_path:
        print(f'Loading unlabeled data from `{args.unlabeled_path}`...')
        with open(args.unlabeled_path) as f:
            unlabeled_data = [replace_brackets(replace_quotes(line.strip().split())) for line in f]

        unlabeled_words = [word for sentence in unlabeled_data for word in sentence]
        words = ptb_words + unlabeled_words
    else:
        words = ptb_words

    if args.lowercase:
        print('Using lowercased vocabulary.')
        words = [word.lower() if word not in BRACKETS else word for word in words]  # do not lowercase brackets

    counts = Counter(words)

    vocabulary = dict((word, count) for word, count in counts.most_common() if count >= args.min_word_count)

    with open(args.vocab_path, 'w') as f:
        json.dump(vocabulary, f, indent=4)

    casing = 'lowercased ' if args.lowercase else 'non-lowercased'
    print(f'Built {casing} vocabulary of size {len(vocabulary)} with minimum count {args.min_word_count}.')
    print(f'Saved vocabulary to `{args.vocab_path}`.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Build a vocabulary from labeled and optional unlabeled data.',
        fromfile_prefix_chars='@')

    parser.add_argument('--train-path', default='data/ptb/train.trees')
    parser.add_argument('--unlabeled-path', default='')
    parser.add_argument('--vocab-path', default='data/vocab/vocab.json')
    parser.add_argument('--min-word-count', type=int, default=1)
    parser.add_argument('--lowercase', action='store_true')

    args = parser.parse_args()

    main(args)
