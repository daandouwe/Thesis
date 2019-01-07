"""
Evaluate accuracy on the syneval dataset.
"""

import os

import dynet as dy
import numpy as np
from tqdm import tqdm

from rnng.decoder import GenerativeDecoder
from utils.general import load_model

FILES = [
    'long_vp_coord',
    # 'npi_across_anim',  # TODO: has three categories
    # 'npi_across_inanim',  # TODO: has three categories
    # 'obj_rel_across_anim',
    # 'obj_rel_across_inanim',
    # 'obj_rel_no_comp_across_anim',
    # 'obj_rel_no_comp_across_inanim',
    # 'obj_rel_no_comp_within_anim',
    # 'obj_rel_no_comp_within_inanim',
    # 'obj_rel_within_anim',
    # 'obj_rel_within_inanim',
    # 'prep_anim',
    # 'prep_inanim',
    # 'reflexive_sent_comp',
    # 'reflexives_across',
    # 'sent_comp',
    'simple_agrmt',
    # 'simple_npi_anim',  # TODO: has three categories
    # 'simple_npi_inanim',  # TODO: has three categories
    'simple_reflexives',
    # 'subj_rel',
    'vp_coord',
]


def syneval_rnn(args):
    model = load_model(args.checkpoint)
    model.eval()

    outpath = os.path.join(args.checkpoint, 'output', 'syneval_results.tsv')

    print('Predicting syneval.')
    print(f'Loading model from `{args.checkpoint}`.')
    print(f'Loading syneval examples from directory `{args.indir}`.')
    print(f'Writing predictions to `{outpath}`.')

    with open(outpath, 'w') as outfile:
        print('\t'.join((
                'name', 'index', 'pos-perplexity', 'neg-perplexity',
                'correct', 'pos-sentence-processed', 'neg-sentence-processed')),
            file=outfile)

        for fname in FILES:
            print(f'Predicting `{fname}`...')

            inpath = os.path.join(args.indir, fname)

            with open(inpath + '.pos') as f:
                pos_sents = [line.strip() for line in f.readlines()]
                if args.capitalize:
                    pos_sents = [sent.capitalize() for sent in pos_sents]
                if args.add_period:
                    pos_sents = [sent + ' .' for sent in pos_sents]

            with open(inpath + '.neg') as f:
                neg_sents = [line.strip() for line in f.readlines()]
                if args.capitalize:
                    neg_sents = [sent.capitalize() for sent in neg_sents]
                if args.add_period:
                    neg_sents = [sent + ' .' for sent in neg_sents]

            pos_sents = [sent.split() for sent in pos_sents]
            neg_sents = [sent.split() for sent in neg_sents]

            assert len(pos_sents) == len(neg_sents)

            correct = 0
            for i, (pos, neg) in enumerate(tqdm(list(zip(pos_sents, neg_sents)))):
                dy.renew_cg()

                pos_pp = np.exp(model.forward(pos).value() / len(pos))
                neg_pp = np.exp(model.forward(neg).value() / len(neg))
                correct += (pos_pp < neg_pp)

                # see which words are unked during prediction
                pos = model.word_vocab.process(pos)
                neg = model.word_vocab.process(neg)

                result =  '\t'.join((
                    fname,
                    str(i),
                    str(round(pos_pp, 2)),
                    str(round(neg_pp, 2)),
                    str(int(pos_pp < neg_pp)),
                    ' '.join(pos),
                    ' '.join(neg)
                ))
                print(result, file=outfile)

            print(f'{fname}: {correct}/{len(pos_sents)} = {correct / len(pos_sents):.2%} correct')


def syneval_rnng(args):
    model = load_model(args.checkpoint)
    proposal = load_model(args.proposal_model)
    decoder = GenerativeDecoder(
        model=model, proposal=proposal, num_samples=args.num_samples)

    outpath = os.path.join(args.checkpoint, 'output', 'syneval_results.tsv')

    print('Predicting syneval.')
    print(f'Loading model from `{args.checkpoint}`.')
    print(f'Loading syneval examples from directory `{args.indir}`.')
    print(f'Writing predictions to `{outpath}`.')

    with open(outpath, 'w') as outfile:
        print('\t'.join((
                'name', 'index', 'pos-perplexity', 'neg-perplexity',
                'correct', 'pos-sentence-processed', 'neg-sentence-processed')),
            file=outfile)

        for fname in FILES:
            print(f'Predicting `{fname}`...')

            inpath = os.path.join(args.indir, fname)

            with open(inpath + '.pos') as f:
                pos_sents = [line.strip() for line in f.readlines()]
                if args.capitalize:
                    pos_sents = [sent.capitalize() for sent in pos_sents]
                if args.add_period:
                    pos_sents = [sent + ' .' for sent in pos_sents]

            with open(inpath + '.neg') as f:
                neg_sents = [line.strip() for line in f.readlines()]
                if args.capitalize:
                    neg_sents = [sent.capitalize() for sent in neg_sents]
                if args.add_period:
                    neg_sents = [sent + ' .' for sent in neg_sents]

            pos_sents = [sent.split() for sent in pos_sents]
            neg_sents = [sent.split() for sent in neg_sents]

            assert len(pos_sents) == len(neg_sents)

            correct = 0
            for i, (pos, neg) in enumerate(tqdm(list(zip(pos_sents, neg_sents)))):
                dy.renew_cg()

                pos_pp = decoder.perplexity(pos)
                neg_pp = decoder.perplexity(neg)
                correct += (pos_pp < neg_pp)

                # see which words are unked during prediction
                pos = model.word_vocab.process(pos)
                neg = model.word_vocab.process(neg)

                result =  '\t'.join((
                    fname,
                    str(i),
                    str(round(pos_pp, 2)),
                    str(round(neg_pp, 2)),
                    str(int(pos_pp < neg_pp)),
                    ' '.join(pos),
                    ' '.join(neg)
                ))
                print(result, file=outfile)

            print(f'{fname}: {correct}/{len(pos_sents)} = {correct / len(pos_sents):.2%} correct')


def main(args):
    if args.model_type == 'gen-rnng':
        syneval_rnng(args)
    elif args.model_type == 'rnn-lm':
        syneval_rnn(args)
