#!/usr/bin/env python

import sys
import re

def simplify(s):
    """Clean nonterminal symbols in s by removing anything beyond the first tag.

    Example: ADJP-PRD=1 --> ADJ
    """
    # Starts with `(` and then any sequence of capital letters: ([A-Z]+
    # we call this chuck nt: (?P<nt>...)
    # then followed by exactly one of `-` or `=`: [-=]{1}
    # and then any number of characters that is not a space: \S
    # and then exactly one space: \s{1}
    # and then keep only nt: \g<nt>
    # NOTE: this is safe, because in the trees no word ever
    # starts with `(` followed by capitals.
    pattern = r'(?P<nt>\([A-Z]+)[-=]{1}\S*\s{1}'
    sub = r'\g<nt> '
    return re.sub(pattern, sub, s)

def main(path):
    with open(path) as f:
        text = f.read()
    text = simplify(text)
    with open(path, 'w') as f:
        print(text, file=f)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        exit('Specify file.')
    main(path)
