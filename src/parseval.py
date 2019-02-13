"""
Custom Parseval implementation.
"""
import argparse

from utils.evalb import Parseval


def main(gold_path, pred_path, tsv_output=False):
    Parseval(gold_path, pred_path).evaluate(tsv_output=tsv_output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PARSEVAL bracketing evaluation.')

    parser.add_argument('gold_path')
    parser.add_argument('pred_path')
    parser.add_argument('--tsv-output', action='store_true')

    args = parser.parse_args()

    main(args.gold_path, args.pred_path, args.tsv_output)
