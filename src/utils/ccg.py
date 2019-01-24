"""
Load and store a PTB annotated with CCG supertags.
"""


class LabeledSequence(object):

    def __init__(self, words, labels):
        self._words = list(words)
        self._labels = list(labels)

    def words(self):
        return self._words

    def labels(self):
        return self._labels


def fromfile(path):
    with open(path) as f:
        sentences = [line for line in f.read().split('\n\n') if line]

        lines = []
        for line in sentences:
            words, labels = zip(*[item.split('\t') for item in line.split('\n')])
            lines.append(LabeledSequence(words, labels))

        return lines


if __name__ == '__main__':
    import os

    home = os.path.expanduser('~')
    thesis = os.path.join(home, 'Documents', 'logic', 'thesis')

    lines = fromfile(os.path.join(thesis, 'data/ccg/train.txt'))

    print(lines[0].words())
    print(lines[0].labels())
