#!/usr/bin/env python

import sys
import re

def add_space(s):
    pattern = r'(?P<nt>.{1})\(' # any `(` preceded by something (i.e. not at start).
    sub = r'\g<nt> (' # put a space beween that something and the bracket.
    return re.sub(pattern, sub, s)

def main(path):
    with open(path) as f:
        text = f.read()
    text = add_space(text)
    with open(path, 'w') as f:
        print(text, file=f)

if __name__ == '__main__':
    # s = '(S(NP(NP(NNP Pierre)(NNP Vinken))(, ,)(ADJP(NP(CD 61)(NNS years))(JJ old))(, ,))(VP(MD will)(VP(VB join)(NP(DT the)(NN board))(PP(IN as)(NP(DT a)(JJ nonexecutive)(NN director)))(NP(NNP Nov.)(CD 29))))(. .))'
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        exit('Specify file.')
    main(path)
