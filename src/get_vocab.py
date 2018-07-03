import argparse
import os

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

def get_vocab(sentences, textline='unked'):
    """Returns the vocabulary used in the oracle file."""
    textline_options = sentences[0].keys()
    assert textline in textline_options, 'invalid choice of textline: choose from {}'.format(list(textline_options))
    vocab = set()
    for sent_dict in sentences:
        vocab.update(set(sent_dict[textline].split()))
    vocab = sorted(list(vocab))
    return vocab

def get_nonterminals(sentences):
    """Returns the set of actions used in the oracle file."""
    nonterminals = set()
    for sent_dict in sentences:
        nts = [a for a in sent_dict['actions'] if a.startswith('NT')]
        nonterminals.update(nts)
    nonterminals = sorted(list(nonterminals))
    return nonterminals

def get_actions(sentences):
    """Returns the set of actions used in the oracle file."""
    actions = set()
    for sent_dict in sentences:
        actions.update(sent_dict['actions'])
    actions = sorted(list(actions))
    return actions


def main(args):
    # Partition the oracle files into sentences
    train = get_sentences(os.path.join(args.oracle_path, 'train', 'ptb.train.oracle'))
    dev = get_sentences(os.path.join(args.oracle_path, 'dev', 'ptb.dev.oracle'))
    test = get_sentences(os.path.join(args.oracle_path, 'test', 'ptb.test.oracle'))
    sentences = train + dev + test

    # Collect desired symbols for our dictionaries
    actions = get_actions(sentences)
    vocab = get_vocab(sentences, textline=args.textline)
    nonterminals = get_nonterminals(sentences)

    # Write out vocabularies
    path = os.path.join(args.oracle_path, 'ptb')
    print('\n'.join(nonterminals),
            file=open(path + '.nonterminals', 'w'))
    print('\n'.join(actions),
            file=open(path + '.actions', 'w'))
    print('\n'.join(vocab),
            file=open(path + '.vocab', 'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data for RNNG parser.')
    parser.add_argument('oracle_path', type=str,
                        help='location of the oracle path')
    parser.add_argument('--textline', type=str, default='unked',
                        help='location of the oracle path')
    args = parser.parse_args()

    main(args)
