"""
Evaluate accuracy on the syneval dataset.
"""

import os
from collections import Counter

import dynet as dy
import numpy as np
from tqdm import tqdm

from rnng.decoder import GenerativeDecoder
from utils.general import load_model


ALL = [
    'simple_agrmt',
    'sent_comp',
    'vp_coord',
    'long_vp_coord',
    'prep_anim',
    'prep_inanim',
    'subj_rel',
    'obj_rel_across_anim',
    'obj_rel_across_inanim',
    'obj_rel_no_comp_across_anim',
    'obj_rel_no_comp_across_inanim',
    'obj_rel_no_comp_within_anim',
    'obj_rel_no_comp_within_inanim',
    'obj_rel_within_anim',
    'obj_rel_within_inanim',

    'simple_reflexives',
    'reflexive_sent_comp',
    'reflexives_across',

    'simple_npi_anim',
    'simple_npi_anim_the',
    'simple_npi_inanim',
    'simple_npi_inanim_the',
    'npi_across_anim',
    'npi_across_anim_the',
    'npi_across_inanim',
    'npi_across_inanim_the',
]


SHORT = [
    'long_vp_coord',
    'simple_agrmt',
    'simple_npi_anim',
    'simple_npi_anim_the',
    'simple_reflexives',
    'vp_coord',
]


def syneval_rnn(args):
    model = load_model(args.checkpoint)
    model.eval()

    files = SHORT if args.syneval_short else ALL

    outdir = os.path.join(args.checkpoint, 'output')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    result_filename = 'syneval_results_caps.tsv' if args.capitalize else 'syneval_results.tsv'
    outpath = os.path.join(outdir, result_filename)

    print('Predicting syneval with RNN language model.')
    print(f'Loading model from `{args.checkpoint}`.')
    print(f'Loading syneval examples from directory `{args.indir}`.')
    print(f'Writing predictions to `{outpath}`.')

    with open(outpath, 'w') as outfile:
        print('\t'.join((
                'name', 'index', 'pos-perplexity', 'neg-perplexity',
                'correct', 'pos-sentence-processed', 'neg-sentence-processed')),
            file=outfile)

        print('Predicting syneval for:', '\n', '\n '.join(files))

        for fname in files:
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

            if args.syneval_max_lines != -1 and len(pos_sents) > args.syneval_max_lines:
                subsampled_ids = np.random.choice(
                    len(pos_sents), args.syneval_max_lines, replace=False)
                pos_sents = [pos_sents[i] for i in subsampled_ids]
                neg_sents = [neg_sents[i] for i in subsampled_ids]

            num_correct = 0
            for i, (pos, neg) in enumerate(tqdm(list(zip(pos_sents, neg_sents)))):
                dy.renew_cg()

                pos_pp = np.exp(model.forward(pos).value() / len(pos))
                neg_pp = np.exp(model.forward(neg).value() / len(neg))
                correct = pos_pp < neg_pp
                num_correct += correct

                # see which words are unked during prediction
                pos = model.word_vocab.process(pos)
                neg = model.word_vocab.process(neg)

                result =  '\t'.join((
                    fname,
                    str(i),
                    str(round(pos_pp, 2)),
                    str(round(neg_pp, 2)),
                    str(int(correct)),
                    ' '.join(pos),
                    ' '.join(neg)
                ))
                print(result, file=outfile)

            print(f'{fname}: {num_correct}/{len(pos_sents)} = {num_correct / len(pos_sents):.2%} correct', '\n')


def syneval_rnng(args):
    model = load_model(args.checkpoint)
    proposal = load_model(args.proposal_model)
    decoder = GenerativeDecoder(
        model=model, proposal=proposal, num_samples=args.num_samples)

    files = SHORT if args.syneval_short else ALL

    outdir = os.path.join(args.checkpoint, 'output')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # result_filename = 'syneval_results_caps.tsv' if args.capitalize else 'syneval_results.tsv'

    # when numpy_seed is not 42 (default) it means we are evaluating
    # the same model repeatedly of different subsamples of the full dataset
    result_filename = f'syneval_results_{args.numpy_seed}.tsv' if args.numpy_seed != 42 else 'syneval_results.tsv'
    outpath = os.path.join(outdir, result_filename)

    print('Predicting syneval with Generative RNNG.')
    print(f'Loading model from `{args.checkpoint}`.')
    print(f'Loading syneval examples from directory `{args.indir}`.')
    print(f'Writing predictions to `{outpath}`.')

    with open(outpath, 'w') as outfile:
        print('\t'.join((
                'name', 'index', 'pos-perplexity', 'neg-perplexity',
                'correct', 'pos-sentence-processed', 'neg-sentence-processed')),
            file=outfile)

        print('Predicting syneval for:', '\n', '\n '.join(files))

        for fname in files:
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

            # subsample the dataset if it exceeds the maximum number of lines specified
            if args.syneval_max_lines != -1 and len(pos_sents) > args.syneval_max_lines:
                subsampled_ids = np.random.choice(
                    len(pos_sents), args.syneval_max_lines, replace=False)
                pos_sents = [pos_sents[i] for i in subsampled_ids]
                neg_sents = [neg_sents[i] for i in subsampled_ids]

            num_correct = 0
            for i, (pos, neg) in enumerate(tqdm(list(zip(pos_sents, neg_sents)))):
                dy.renew_cg()

                pos_pp = decoder.perplexity(pos)
                neg_pp = decoder.perplexity(neg)
                correct = pos_pp < neg_pp
                num_correct += correct

                # see which words are unked during prediction
                pos = model.word_vocab.process(pos)
                neg = model.word_vocab.process(neg)

                result =  '\t'.join((
                    fname,
                    str(i),
                    str(round(pos_pp, 2)),
                    str(round(neg_pp, 2)),
                    str(int(correct)),
                    ' '.join(pos),
                    ' '.join(neg)
                ))
                print(result, file=outfile)

            print(f'{fname}: {num_correct}/{len(pos_sents)} = {num_correct / len(pos_sents):.2%} correct', '\n')


def syneval_parser(args):
    model = load_model(args.checkpoint)

    files = SHORT if args.syneval_short else ALL

    outdir = os.path.join(args.checkpoint, 'output')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    result_filename = 'syneval_results_caps.tsv' if args.capitalize else 'syneval_results.tsv'
    outpath = os.path.join(outdir, result_filename)

    print('Predicting syneval with discriminative parser.')
    print(f'Loading model from `{args.checkpoint}`.')
    print(f'Loading syneval examples from directory `{args.indir}`.')
    print(f'Writing predictions to `{outpath}`.')

    with open(outpath, 'w') as outfile:
        if self.num_samples == 1:
            # predict with logprob of predicted parse
            print('\t'.join((
                    'name', 'index', 'pos-logprob', 'neg-logprob',
                    'correct', 'pos-tree', 'neg-tree')),
                file=outfile)
        else:
            print('\t'.join((
                    'name', 'index', 'pos-entropy', 'neg-entropy',
                    'correct', 'pos-tree', 'neg-tree')),
                file=outfile)

        print('Predicting syneval for:', '\n', '\n '.join(files))

        for fname in files:
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

            if args.syneval_max_lines != -1 and len(pos_sents) > args.syneval_max_lines:
                subsampled_ids = np.random.choice(
                    len(pos_sents), args.syneval_max_lines, replace=False)
                pos_sents = [pos_sents[i] for i in subsampled_ids]
                neg_sents = [neg_sents[i] for i in subsampled_ids]

            num_correct = 0
            for i, (pos, neg) in enumerate(tqdm(list(zip(pos_sents, neg_sents)))):
                dy.renew_cg()

                if args.num_samples == 1:
                    # predict with logprob of predicted parse
                    pos_tree, pos_nll = model.parse(pos)
                    neg_tree, neg_nll = model.parse(neg)

                    pos_logprob = -pos_nll.value()
                    neg_logprob = -neg_nll.value()

                    pos_tree = pos_tree.linearize(with_tag=False)
                    neg_tree = neg_tree.linearize(with_tag=False)

                    correct = pos_logprob > neg_logprob
                    num_correct += correct

                    result =  '\t'.join((
                        fname,
                        str(i),
                        str(round(pos_logprob, 4)),
                        str(round(neg_logprob, 4)),
                        str(int(correct)),
                        pos_tree,
                        neg_tree
                    ))
                    print(result, file=outfile)

                else:
                    # predict with the parser's entropy
                    pos_trees, pos_nlls = zip(*
                        [model.sample(pos) for _ in range(args.num_samples)])
                    neg_trees, neg_nlls = zip(*
                        [model.sample(neg) for _ in range(args.num_samples)])

                    pos_entropy = np.mean([nll.value() for nll in pos_nlls])
                    neg_entropy = np.mean([nll.value() for nll in neg_nlls])

                    pos_tree = Counter(
                        [tree.linearize(with_tag=False) for tree in pos_trees]).most_common(1)[0][0]
                    neg_tree = Counter(
                        [tree.linearize(with_tag=False) for tree in neg_trees]).most_common(1)[0][0]

                    correct = pos_entropy < neg_entropy
                    num_correct += correct

                    result =  '\t'.join((
                        fname,
                        str(i),
                        str(round(pos_entropy, 4)),
                        str(round(neg_entropy, 4)),
                        str(int(correct)),
                        pos_tree,
                        neg_tree
                    ))
                    print(result, file=outfile)

            print(f'{fname}: {num_correct}/{len(pos_sents)} = {num_correct / len(pos_sents):.2%} correct', '\n')


def main(args):
    if args.model_type == 'gen-rnng':
        syneval_rnng(args)
    elif args.model_type == 'rnn-lm':
        syneval_rnn(args)
    elif args.model_type in ('crf', 'disc-rnng'):
        syneval_parser(args)
    else:
        exit('invalid model type specified')
