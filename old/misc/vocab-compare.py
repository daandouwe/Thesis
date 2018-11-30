#!/usr/bin/env python
"""
Compare vocabularies of the PTB and Wikitext.
"""
import os


def get_wiki_vocab(wikidir):
    with open(wikidir) as f:
        text = f.read()

    with open('wiki-vocab.lower.txt', 'w') as out:
        vocab = set(text.lower().split())
        print(len(vocab), file=out)
        print('\n'.join(sorted(vocab)), file=out)

    with open('wiki-vocab.txt', 'w') as out:
        vocab = set(text.split())
        print(len(vocab), file=out)
        print('\n'.join(sorted(vocab)), file=out)


def main():
    wikidir = os.path.expanduser('~/data/wikitext/wikitext-2/wiki.train.tokens')
    wiki_vocab_path = 'wiki-vocab.lower.txt'
    ptb_vocab_path = '../data/vocab/unked/ptb.vocab'

    get_wiki_vocab(wikidir)

    with open(ptb_vocab_path) as f:
        ptb_vocab = set(line.strip() for line in f)

    with open(wiki_vocab_path) as f:
        wiki_vocab = set(line.strip() for line in f)

    wiki_or_ptb_vocab = (wiki_vocab | ptb_vocab)
    wiki_and_ptb_vocab = (wiki_vocab & ptb_vocab)
    wiki_not_ptb = wiki_vocab - ptb_vocab
    ptb_not_wiki = ptb_vocab - wiki_vocab

    with open('wiki_or_ptb_vocab.txt', 'w') as f:
        print('wiki', len(wiki_vocab), 'ptb', len(ptb_vocab), 'either', len(wiki_or_ptb_vocab), file=f)
        print('\n'.join(sorted(wiki_or_ptb_vocab)), file=f)

    with open('wiki_and_ptb_vocab.txt', 'w') as f:
        print('wiki', len(wiki_vocab), 'ptb', len(ptb_vocab), 'both', len(wiki_and_ptb_vocab), file=f)
        print('\n'.join(sorted(wiki_and_ptb_vocab)), file=f)

    with open('wiki_not_ptb.txt', 'w') as f:
        print('wiki', len(wiki_vocab), 'ptb', len(ptb_vocab), 'not ptb', len(wiki_not_ptb), file=f)
        print('\n'.join(sorted(wiki_not_ptb)), file=f)

    with open('ptb_not_wiki.txt', 'w') as f:
        print('wiki', len(wiki_vocab), 'ptb', len(ptb_vocab), 'not wiki', len(ptb_not_wiki), file=f)
        print('\n'.join(sorted(ptb_not_wiki)), file=f)



if __name__ == '__main__':
    main()
