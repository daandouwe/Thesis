#!/usr/bin/env python
"""
Process a section of the one-billion-word dataset (http://www.statmt.org/lm-benchmark/)
to make the tokenization consistent with the Penn Treebank, and control the size:
    1) Replace brackets with their PTB-style escapes: '(' -> '-LRB-' etc.
    2) Replace quotation marks with their PTB equivalents: " -> `` etc.
    3) Select only sentences shorter than MAX_LEN words.
    4) Select only MAX_LINES number of sentence.
"""

import sys
from collections import Counter


MAX_LEN = 40
MAX_LINES = 100000

BRACKETS = [
    '-LRB-', '-RRB-',
    '-LSB-', '-RSB-',
    '-RCB-', '-RCB-'
]


def replace_quotes(words):
    """Replace quotes following PTB convention"""
    assert isinstance(words, list), words
    assert all(isinstance(word, str) for word in words), words

    replaced = []
    found_left_double, found_left_single = False, False
    for word in words:
        if word == '"':
            if found_left_double:
                found_left_double = False
                replaced.append("''")
            else:
                found_left_double = True
                replaced.append("``")
        elif word == "'":
            if found_left_double:
                found_left_double = False
                replaced.append("'")
            else:
                found_left_double = True
                replaced.append("`")
        else:
            replaced.append(word)
    return replaced


def replace_brackets(words):
    """Replace brackets following PTB convention"""
    assert isinstance(words, list), words
    assert all(isinstance(word, str) for word in words), words

    replaced = []
    for word in words:
        if word == '(':
            replaced.append('-LRB-')
        elif word == '{':
            replaced.append('-LCB-')
        elif word == '[':
            replaced.append('-LSB-')
        elif word == ')':
            replaced.append('-RRB-')
        elif word == '}':
            replaced.append('-RCB-')
        elif word == ']':
            replaced.append('-RSB-')
        else:
            replaced.append(word)
    return replaced


def main(inpath):
    with open(inpath) as f:
        lines = [replace_brackets(replace_quotes(line.strip().split()))
            for line in f.readlines()]

    lines = [' '.join(line) for line in lines if len(line) < MAX_LEN][:MAX_LINES]

    print('\n'.join(lines))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit('Specify input path.')
    else:
        main(sys.argv[1])
