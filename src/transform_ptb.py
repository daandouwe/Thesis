##################################################################
# Transform the Penn Treebank from a collections of folders with
# mrg files into one long document with linearized parse trees,
# one sentence per line. Prints to stdout.
##################################################################

import re
import os

def partition(sent, indices):
    parts = []
    for start, end in zip(indices, indices[1:]+[len(sent)]):
        parts.append(sent[start:end])
    return parts

def transform_mrg(path):
    """Cleans the mrg file. Returns a one-line string in the format
    given by sample_input_english.txt
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
    for path in ptb_folders_iter(corpus_root):
        transform_mrg(path)


if __name__ == '__main__':
    corpus_root = '../data/ptb/con/treebank3/parsed/mrg/wsj'
    file_pattern = r'.*/wsj_.*\.mrg'
    file_path = corpus_root + '/00/wsj_0002.mrg'
    # transform_mrg(file_path)
    main()
