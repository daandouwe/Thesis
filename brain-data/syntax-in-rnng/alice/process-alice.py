#!/usr/bin/env python
import argparse
import re
import string
import warnings; warnings.filterwarnings("ignore")  # spaCy gives warnings that can be ignored

from gutenberg import acquire, cleanup
import spacy
from spacy.lang.en import English
import nltk
nltk.download('punkt')

ALICE = 11  # Project Gutenberg file-code for `alice in wonderland`


def heuristic_sentence_cleanup(sentences):
    """Cleans up strange cases with some heuristics."""
    symbols = [
        ('(', ')'),
        ('``', "''"),
        ('`', "'")
    ]

    def remove_balance(string, symbols):
        for lsymbol, rsymbol in symbols:
            if string.count(lsymbol) == string.count(rsymbol) == 1:
                # 'This happens to be a citation.' --> This happens to be a citation.
                string = string[len(lsymbol):-len(rsymbol)]
        return string

    sentences = [remove_balance(sentence, symbols) for sentence in sentences]
    return sentences


def normalize_citation_marks(text):
    """Normalize apotrophes and citation marks following PTB conventions.

    Examples:
    ‘...’ --> `...'
    “...” --> ``...''
    “‘...’” --> `` `...' ''

    Source:
    >>> import nltk
    >>> nltk.help.upenn_tagset()
    ...
    '': closing quotation mark
        ' ''
    ``: opening quotation mark
        ` ``
    """
    text = re.sub(r'’', r"'", text)
    text = re.sub(r'‘', r'`', text)
    text = re.sub(r'”', r"'' ", text)
    text = re.sub(r'“', r' ``', text)
    return text


def remove_punct(sentences):
    """Sometimes you want no punctuation at all."""
    # remove all punct except final punct and end-quote
    sentences = [re.sub(r'[`,;:]', r'', sent) for sent in sentences]
    # remove em-dash
    sentences = [re.sub(r'--', r'', sent) for sent in sentences]
    # remove ' if end-quote (but not in `can't`!)
    sentences = [re.sub(r"''", r'', sent) for sent in sentences]
    sentences = [re.sub(r"'(?!\S)", r'', sent) for sent in sentences]
    return sentences


def main(args):
    nlp = spacy.load('en')
    spacy_tokenizer = English().Defaults.create_tokenizer(nlp)

    # Read Gutenberg text.
    text = cleanup.strip_headers(acquire.load_etext(ALICE)).strip()
    # Remove newline inside a sentence `the\nbank` --> `the bank`
    text = re.sub(r'(?P<a>\S+)\n{1}(?P<b>\S+)', '\g<a> \g<b>', text)
    # Get rid of variety in citation marks.
    text = normalize_citation_marks(text)
    # The one case _I_ --> I
    text = re.sub(r'_', r'', text)
    # Chunk into chapters.
    chapters = text.split('\n\n\n\n\n')
    for i, chapter in enumerate(chapters):
        # Remove chapter heading.
        chapter = re.sub('^CHAPTER.*\n+(?P<rest>\S+)', '\g<rest>', chapter)
        # Replace all whitespace by a single space.
        chapter = re.sub('\s+', ' ', chapter)
        # Use nltk to chop chapter into sentences.
        sentences = nltk.tokenize.sent_tokenize(chapter)
        # Remove weird * * * stuff from chapter 1
        sentences = [sentence.strip() for sentence in sentences if not sentence.startswith('* *')]
        if args.remove_punct:
            sentences = remove_punct(sentences)
        if args.heuristic_cleanup:
            sentences = heuristic_sentence_cleanup(sentences)
        # Print raw cleaned sentence.
        with open(f'alice.{i}.txt', 'w') as f:
            for sentence in sentences:
                print(sentence, file=f)
        # Print spacy tokenization.
        with open(f'alice.{i}.toks', 'w') as f:
            for sentence in sentences:
                tokens = spacy_tokenizer(sentence)
                sentence = ' '.join([token.text for token in tokens])
                print(sentence, file=f)
        # Print nltk tokenization.
        with open(f'alice.{i}.tokn', 'w') as f:
            for sentence in sentences:
                tokens = nltk.tokenize.word_tokenize(sentence)
                sentence = ' '.join(tokens)
                print(sentence, file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processing for `Alice in Wonderland`.')
    parser.add_argument('--remove-punct', action='store_true')
    parser.add_argument('--heuristic-cleanup', action='store_true')
    args = parser.parse_args()

    main(args)
