#!/usr/bin/env python
import re
import numpy as np
import warnings; warnings.filterwarnings("ignore")

import benepar
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def escape_brackets(text):
    """Escape brackets following PTB conventions.

    Examples:
    `() ))` --> `(-RRB- -RRB-)`
    `(( {)` --> `(-LRB- -LCB-)`

    Source:
    >>> import nltk
    >>> nltk.help.upenn_tagset()
    ...
    (: opening parenthesis
        ( [ {
    ): closing parenthesis
        ) ] }
    """
    old_left_tag, old_right_tag = r'\(', r'\)'  # escape for use in regex
    new_left_tag, new_right_tag = '-LRB-', '-RRB-'
    left_brackets = {
        r'\(': '-LRB-',  # escape for use in regex
        r'{': '-LCB-',
        r'\[': '-LSB-',  # escape for use in regex
    }
    right_brackets = {
        r'\)': '-RRB-',  # escape for use in regex
        r'}': '-RCB-',
        r']': '-RSB-',
    }
    re_leafnode = '\({} {}\)'  # escape for use in regex
    pr_leafnode = '({} {})'  # for printing
    for bracket, token in left_brackets.items():
        old = re_leafnode.format(old_left_tag, bracket)
        new = pr_leafnode.format(new_left_tag, token)
        text = re.sub(old, new, text)
    for bracket, token in right_brackets.items():
        old = re_leafnode.format(old_right_tag, bracket)
        new = pr_leafnode.format(new_right_tag, token)
        text = re.sub(old, new, text)
    return text


def main():
    parser = benepar.Parser("benepar_en")
    print('Parsing...')
    for i in range(13):
        trees = []
        with open(f'alice.{i}.tokn') as f:
            for j, line in enumerate(f):
                print(f'Chapter {i:>2} line {j:>4}.', end='\r')
                tree = parser.parse(line.strip())
                tree_string = tree.pformat(margin=np.inf)
                trees.append(tree_string)
        trees = '\n'.join(trees)
        # Escape brackets _after_ parsing because benepar expects brackets in text.
        trees = escape_brackets(trees)
        with open(f'alice.{i}.trees', 'w') as f:
            print(trees, file=f)
    print('Finished parsing.')


if __name__ == '__main__':
    main()
