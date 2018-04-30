##################################################################
# Transform the Penn Treebank from a collections of folders with
# mrg files into one long document with linearized parse trees,
# one sentence per line. Prints to stdout.
##################################################################

import os
import sys
import re

def partition(sent, indices):
    parts = []
    for start, end in zip(indices, indices[1:]+[len(sent)]):
        parts.append(sent[start:end])
    return parts

def transform_mrg(path):
    """Cleans the mrg file. Prints a one-line string to stdout in the
    format given by sample_input_english.txt
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
            print(line[1:-1])

def ptb_folders_iter(corpus_root):
    """Returns an iterator over all paths to the .mrg files
    in the wsj part of the ptb.
    """
    for subdir, dirs, files in os.walk(corpus_root):
        for file in files:
            if file.endswith('.mrg'):
                yield(os.path.join(subdir, file))

def main():
    assert len(sys.argv) > 1, 'Specify the directory to the WSJ.'
    corpus_root = sys.argv[1] # '../data/ptb/con/treebank3/parsed/mrg/wsj'
    if len(sys.argv) > 2:
        nlines = int(sys.argv[2])
    else:
        nlines = None
    for i, path in enumerate(ptb_folders_iter(corpus_root)):
        if nlines is not None and i > nlines:
            break
        transform_mrg(path)

if __name__ == '__main__':
    main()
