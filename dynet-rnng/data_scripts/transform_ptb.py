"""Transform the Penn Treebank from a collections of folders with
mrg files into one long document with linearized parse trees,
one sentence per line.
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
    """Clean an mrg file manually.

    Turns each of the fancy formatted trees in the mrg
    file into one-line strings, in the format given by:
    (S (NP (DT The) (NN cat)) (VP (VBZ sleaps))) (. .))
    """
    with open(path) as fin:
        s = fin.read()
        s = re.sub('\n', '', s)    # put string onto one long line
        s = re.sub(' +', ' ', s)   # remove excess whitespace (indentation)
        s = re.sub('\( ', '(', s)  # `( (S ... -> ((S`
        s = re.sub(' \)', ')', s)  # `(NN director) ))` --> `(NN director)))`
        bounds = [m.start() for m in re.finditer('\(\(', s)]  # each tree starts with double left brackets.
        trees = partition(s, bounds)
        for tree in trees:
            yield tree[1:-1]  # `((S ... (. .)))` --> `(S ... (. .))`

def transform_mrg_nltk(path):
    """Let NTLK do the dirty work."""
    from nltk.corpus import ptb  # TODO: move these imports
    from numpy import inf
    for tree in ptb.parsed_sents(path):
        yield tree.pformat(margin=inf)  # format nltk tree on one line (no margin to indent at)

def ptb_folders_iter(corpus_root):
    """Iterator over all mrg filepaths in the wsj part of the ptb."""
    train_folders = ['00', '01', '02', '03', '04', '05', '06', '07',
                     '08', '09', '10', '11', '12', '13', '14', '15',
                     '16', '17', '18', '19', '20', '21', '22']
    dev_folder = '23'
    test_folder = '24'
    def train():
        for dir in train_folders:
            path = os.path.join(corpus_root, dir)
            for file in os.listdir(path):
                if file.endswith('.mrg'):
                    yield os.path.join(path, file)
    def dev():
        path = os.path.join(corpus_root, dev_folder)
        for file in os.listdir(path):
            if file.endswith('.mrg'):
                yield os.path.join(path, file)
    def test():
        path = os.path.join(corpus_root, test_folder)
        for file in os.listdir(path):
            if file.endswith('.mrg'):
                yield os.path.join(path, file)
    return train(), dev(), test()

def main(args):
    indir = os.path.expanduser(args.indir)  # replace `~` with $HOME
    train, dev, test = ptb_folders_iter(indir)

    train_path = os.path.join(args.outdir, 'train', args.name + '.train.trees')
    with open(train_path, 'w') as f:
        for path in train:
            lines = transform_mrg(path)
            for line in lines:
                print(line, file=f)

    dev_path = os.path.join(args.outdir, 'dev', args.name + '.dev.trees')
    with open(dev_path, 'w') as f:
        for path in dev:
            lines = transform_mrg(path)
            for line in lines:
                print(line, file=f)

    test_path = os.path.join(args.outdir, 'test', args.name + '.test.trees')
    with open(test_path, 'w') as f:
        for path in test:
            lines = transform_mrg(path)
            for line in lines:
                print(line, file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='~/data/ptb/con/treebank3/parsed/mrg/wsj',
                        help='the directory to the PTB')
    parser.add_argument('--outdir', type=str, default='../tmp',
                        help='path to write the transformed mrg files')
    parser.add_argument('--name', type=str, default='ptb',
                        help='name the file')
    args = parser.parse_args()

    main(args)
