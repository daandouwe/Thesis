#!/usr/bin/env python
import sys
import re

def simplify(s):
    """Clean nonterminal symbols in s by removing anything beyond the first tag.

    Example: ADJP-PRD=1 --> ADJ
    """
    # Nonterminal starts with `(` and then any sequence of capital letters: ([A-Z]+
    # and escape bracket: \(. We call this chunk nt: (?P<nt>...)
    # then followed by exactly one of `-` or `=` or `|`: [-=|]{1}
    # then any number of characters that is not a whitespace: \S
    # and then exactly one space: \s{1}
    # We then keep only nt: \g<nt>
    # Note: this is safe, because in the sentences no word ever
    # starts with `(` followed by capitals.
    pattern = r'(?P<nt>\([A-Z]+)[-=|]{1}\S+\s{1}'
    sub = r'\g<nt> '
    return re.sub(pattern, sub, s)

def main(path):
    with open(path) as f:
        text = f.read()
    text = simplify(text)
    with open(path, 'w') as f:
        print(text, file=f, end='')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        exit('Specify file.')
    main(path)
