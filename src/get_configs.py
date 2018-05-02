import sys

EMPTY_TOKEN = '-EMPTY-'
SEPARATOR = ' || '

def get_actions(path):
    """Returns the set of actions used in the oracle file."""
    actions = {'SHIFT', 'REDUCE'}
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('NT('):
                actions.add(line)
    actions = sorted(list(actions))
    return actions

def get_vocab(sentences):
    """Returns the vocabulary used in the oracle file."""
    vocab = set()
    for sent in sentences:
        vocab.update(set(sent[4].split()))
    vocab = sorted(list(vocab))
    return vocab

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

def get_configurations(sent, separator=SEPARATOR):
    tree, tags, words = sent[0].split(), sent[1].split(), sent[4].split()
    actions = sent[5:]
    # First configuration
    prev_a = actions[0]
    stack = []
    buffer = words
    print(EMPTY_TOKEN, ' '.join(buffer), EMPTY_TOKEN, prev_a, sep=separator)
    # Rest of the configurations
    for i, a in enumerate(actions[1:], 1):
        if prev_a == 'SHIFT':
            # move first element of buffer to stack
            stack.append(buffer[0])
            buffer = buffer[1:]
            # start of the sentence
            if buffer == []:
                buffer = [EMPTY_TOKEN]
        elif prev_a == 'REDUCE':
             # close bracket
            stack.append(')')
        elif prev_a.startswith('NT('):
            # select `X` from `NT(X)`
            stack.append(prev_a[2:-1])
        else:
            raise NotImplementedError('Caught illegitimate action: {}'.format(prev_a))
        # print the configuration
        print(' '.join(stack), ' '.join(buffer), ' '.join(actions[:i]), a,
                sep=separator)
        prev_a = a
    return stack


def main():
    assert len(sys.argv) > 1, 'Specify an oracle file.'
    oracle_path = sys.argv[1]
    sentences = chunk_oracle_file(oracle_path)

    stack_symbols = set()
    for sent in sentences:
        # prints the configurations, and returns the final stack content
        stack = get_configurations(sent)
        stack_symbols.update(set(stack))

    stack_symbols = sorted(list(stack_symbols))
    vocab = get_vocab(sentences)
    actions = get_actions(oracle_path)
    print('\n'.join(stack_symbols), file=open('../tmp/ptb.stack', 'w'))
    print('\n'.join(actions), file=open('../tmp/ptb.actions', 'w'))
    print('\n'.join(vocab), file=open('../tmp/ptb.vocab', 'w'))


if __name__ == '__main__':
    main()
