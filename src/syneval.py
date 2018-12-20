"""
Evaluate accuracy on the syneval dataset.
"""

import os

import dynet as dy
from tqdm import tqdm
import numpy as np

from rnng.decoder import GenerativeDecoder


def load_model(dir):
    model = dy.ParameterCollection()
    [parser] = dy.load(dir, model)
    return parser


def main(args):

    model = load_model(args.checkpoint)
    proposal = load_model(args.proposal_model)
    decoder = GenerativeDecoder(
        model=model, proposal=proposal, num_samples=args.num_samples)

    with open(args.infile + '.pos') as f:
        pos_sents = [line.strip().split() for line in f.readlines()]

    with open(args.infile + '.neg') as f:
        neg_sents = [line.strip().split() for line in f.readlines()]

    correct = 0
    with open(args.outfile, 'w') as f:
        for i, (pos, neg) in enumerate(zip(pos_sents, neg_sents)):
            pos_pp = decoder.perplexity(pos)
            neg_pp = decoder.perplexity(neg)
            correct += (pos_pp < neg_pp)

            pos = model.word_vocab.process(pos)
            neg = model.word_vocab.process(neg)

            result =  '|||'.join(
                    i, round(pos_pp, 2), round(neg_pp, 2), pos_pp < neg_pp, correct, ' '.join(neg), ' '.join(pos)
                )
            print(result)
            print(result, file=f)

    print(f'Syneval: {correct}/{len(pos_sents)} = {correct / len(pos_sents):%} correct')
