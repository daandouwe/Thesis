import sys

import torch

EMPTY_TOKEN = '<EMPTY>'

def get_actions(path):
    """Returns the set of actions used in the oracle file."""
    actions = {'SHIFT', 'REDUCE'}
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('NT('):
                actions.add(line)
    return actions

def chunk_oracle_file(path):
    sentences = []
    with open(path) as f:
        sent = []
        for line in f:
            if line == '\n':
                sentences.append(sent)
                sent = []
            else:
                sent.append(line.rstrip())
        return sentences

def get_batches(sent, separator=' || '):
    tree, tags, words, actions = sent[0].split(), sent[1].split(), sent[2].split(), sent[5:]
    # First configuration
    prev_a = actions[0]
    stack = []
    buffer = words
    print(EMPTY_TOKEN, ' '.join(buffer), prev_a, sep=separator)
    # Rest of the configurations
    for a in actions[1:]:
        if prev_a == 'SHIFT':
            stack.append(buffer[0])
            buffer = buffer[1:]
            if buffer == []:
                buffer = [EMPTY_TOKEN]
            print(' '.join(stack), ' '.join(buffer), a, sep=separator)
        elif prev_a == 'REDUCE':
            stack.append(')')
            print(' '.join(stack), ' '.join(buffer), a, sep=separator)
        elif prev_a.startswith('NT('):
            stack.append(prev_a[2:-1]) # select `X` from `NT(X)`
            print(' '.join(stack), ' '.join(buffer), a, sep=separator)
        else:
            raise NotImplementedError('Caught illegitimate action: {}'.format(prev_a))
        prev_a = a
    print()


def main():
    assert len(sys.argv) > 1, 'Specify an oracle file.'
    oracle_path = sys.argv[1]
    # actions = get_actions(oracle_path)
    sentences = chunk_oracle_file(oracle_path)
    for sent in sentences:
        get_batches(sent)

if __name__ == '__main__':
    main()
