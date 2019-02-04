"""
Custom Parseval implementation.
"""

import sys

from utils.evalb import Parseval


def main(gold_path, pred_path):
    Parseval(gold_path, pred_path).evaluate()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        exit('Usage: \n  python parseval.py gold_path pred_path')

    main(sys.argv[1], sys.argv[2])
