import argparse

def get_sent_dict(sent):
    """Organize a sentence from the oracle file  as a dictionary."""
    d = {
        'tree'    : sent[0],
        'tags'    : sent[1],
        'upper'   : sent[2],
        'lower'   : sent[3],
        'unked'   : sent[4],
        'actions' : sent[5:]
        }
    return d

def get_sentences(path):
    """Chunks the oracle file into sentences.

    Returns:
        A list of sentences. Each sentence is dictionary as returned by
        get_sent_dict.
    """
    sentences = []
    with open(path) as f:
        sent = []
        for line in f:
            if line == '\n':
                sentences.append(sent)
                sent = []
            else:
                sent.append(line.rstrip())
        # sentences is of type [[str]]
        return [get_sent_dict(sent) for sent in sentences]

def get_actions(sentences):
    """Returns the set of actions used in the oracle file."""
    actions = set()
    for sent_dict in sentences:
        actions.update(sent_dict['actions'])
    actions = sorted(list(actions))
    return actions

def get_vocab(sentences, text_line='unked'):
    """Returns the vocabulary used in the oracle file."""
    vocab = set()
    for sent_dict in sentences:
        vocab.update(set(sent_dict[text_line].split()))
    vocab = sorted(list(vocab))
    return vocab

def main(args):
    # Partition the oracle file into sentences
    sentences = get_sentences(args.oracle_path)

    # Collect desired symbols for our dictionaries
    actions = get_actions(sentences)
    vocab = get_vocab(sentences)
    stack = actions + vocab

    # Write out vocabularies
    path, extension = args.oracle_path.split('.oracle')
    print('\n'.join(stack),
            file=open(path + '.stack', 'w'))
    print('\n'.join(actions),
            file=open(path + '.actions', 'w'))
    print('\n'.join(vocab),
            file=open(path + '.vocab', 'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data for RNNG parser.')
    parser.add_argument('oracle_path', type=str, help='location of the oracle path')

    args = parser.parse_args()

    main(args)
