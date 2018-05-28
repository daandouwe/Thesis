"""
Transform the Penn Treebank from a collections of folders with
mrg files into one long document with linearized parse trees,
one sentence per line. Prints to stdout.
"""

import os
import re
import argparse

def partition(sent, indices):
    parts = []
    for start, end in zip(indices, indices[1:]+[len(sent)]):
        parts.append(sent[start:end])
    return parts

def transform_mrg(path):
    """Clean the mrg file.

    Prints a one-line string to stdout in the format given
    by sample_input_english.txt
    """
    with open(path) as s:
        s = s.read()
        s = re.sub('\n', '', s)
        s = re.sub(' +', ' ', s)
        s = re.sub('\( ', '(', s)
        s = re.sub(' \)', ')', s)
        bounds = [m.start() for m in re.finditer('\(\(', s)]
        parts = partition(s, bounds)
        for line in parts:
            return line[1:-1]

def ptb_folders_iter(corpus_root):
    """Iterator over all mrg filepaths in the wsj part of the ptb.

    # TODO: edit the iterator to perform stadard train/dev/test splits
    """
    for subdir, dirs, files in os.walk(corpus_root):
        for file in files:
            if file.endswith('.mrg'):
                yield(os.path.join(subdir, file))

def main(args):
    for i, path in enumerate(ptb_folders_iter(args.path)):
        if args.nlines > -1: # If we put a maximum on the number of lines
            if i > args.nlines:
                break
        line = transform_mrg(path)
        print(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str,
                        help='the directory to the PTB')
    parser.add_argument('--nlines', type=int, default=-1,
                        help='maximum number of lines to process')
    args = parser.parse_args()

    main(args)
