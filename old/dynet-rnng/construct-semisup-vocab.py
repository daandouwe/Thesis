import argparse
import json
from collections import Counter

from tree import fromstring
from utils import replace_brackets, replace_quotes

# special ptb escapes for brackets
BRACKETS = [
    'LRB', 'RRB',
    'LSB', 'RSB',
    'RCB', 'RCB'
]

def main(args):
    with open(args.tree_path) as f:
        ptb_data = [fromstring(line.strip()) for line in f]

    with open(args.unlabeled_path) as f:
        unlabeled_data = [replace_brackets(replace_quotes(line.strip().split())) for line in f]

    ptb_words = [word for tree in ptb_data for word in tree.leaves()]
    unlabeled_words = [word for sentence in unlabeled_data for word in sentence]

    if args.lowercase:
        ptb_words = [word.lower() if word not in BRACKETS else word for word in ptb_words]
        unlabeled_words = [word.lower() if word not in BRACKETS else word for word in unlabeled_words]

    counts = Counter(ptb_words + unlabeled_words)

    vocab = dict((word, count) for word, count in counts.most_common() if count >= args.min_count)

    with open(args.out_path, 'w') as f:
        json.dump(vocab, f, indent=4)

    print(f'Saved vocabulary to `{args.out_path}`.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--tree-path', default='../data/train/ptb.train.trees')
    parser.add_argument('--unlabeled-path', default='../data/unlabeled/news.en-00001-of-00100')
    parser.add_argument('--out-path', default='../data/vocabs/semisup-vocab.json')
    parser.add_argument('--min-count', type=int, default=2)
    parser.add_argument('--lowercase', action='store_true')

    args = parser.parse_args()

    main(args)
