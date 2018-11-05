#!/usr/bin/env python
import os
import glob
import tempfile
import multiprocessing as mp

from tqdm import tqdm
from nltk import Tree

from decode import (GreedyDecoder, SamplingDecoder,
    GenerativeImportanceDecoder, GenerativeSamplingDecoder)
from eval import evalb
from utils import ceil_div


def remove_duplicates(samples):
    """Filter out duplicate trees from the samples."""
    output = []
    seen = set()
    for tree, proposal_logprob, logprob in samples:
        if tree.linearize() not in seen:
            output.append((tree, proposal_logprob, logprob))
            seen.add(tree.linearize())
    return output


def get_checkfile(checkpoint):
    if not checkpoint:
        latest_dir = max(glob.glob(os.path.join('checkpoints', '*/')))
        return os.path.join(latest_dir, 'model.pt')
    else:
        return checkpoint


def is_tree(line):
    """Simple `oracle` to see if line is a tree."""
    assert isinstance(line, str), line
    try:
        Tree.fromstring(line)
        return True
    except ValueError:
        return False


def predict_file(args):
    assert os.path.exists(args.data), 'specifiy file to parse with --data.'
    print(f'Predicting trees for lines in `{args.data}`.')
    with open(args.data, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    if is_tree(lines[0]):
        lines = [Tree.fromstring(line).leaves() for line in lines]

    checkfile = get_checkfile(args.checkpoint)
    if args.rnng_type == 'disc':
        print('Predicting with discriminative model.')
        decoder = GreedyDecoder(use_tokenizer=False)
        decoder.load_model(path=checkfile)
    elif args.rnng_type == 'gen':
        print('Predicting with generative model.')
        decoder = GenerativeImportanceDecoder(use_tokenizer=False)
        decoder.load_model(path=checkfile)
        if args.proposal_model:
            decoder.load_proposal_model(path=args.proposal_model)
        if args.proposal_samples:
            decoder.load_proposal_samples(path=args.proposal_samples)

    print(f'Predicting trees for `{args.data}`...')
    if args.num_procs == 1:
        trees = []
        for line in tqdm(lines):
            tree, *rest = decoder(line)
            trees.append(tree)
    else:
        tuples = decoder.decode_parallel(lines, num_procs=args.num_procs, with_tag=True)  # (tree, logprob) tuples
        trees, _ = zip(*tuples)

    # Make a temporay directory for the EVALB files.
    pred_path = os.path.join(args.outdir, 'predicted.txt')
    gold_path = os.path.join(args.outdir, 'gold.txt')
    result_path = os.path.join(args.outdir, 'output.txt')
    # Save the predicted trees.
    with open(pred_path, 'w') as f:
        print('\n'.join(trees), file=f)
    # Also save the gold trees in the temp dir for easy inspection.
    with open(args.data, 'r') as fin:
        with open(gold_path, 'w') as fout:
            print(fin.read(), file=fout, end='')
    # Score the trees.
    fscore = evalb(args.evalb_dir, pred_path, gold_path, result_path)
    print(f'Finished. F-score {fscore:.2f}. Results saved in `{args.outdir}`.')


def predict_input_disc(args):
    print('Predicting with discriminative model.')
    greedy = GreedyDecoder(use_tokenizer=args.use_tokenizer)
    checkfile = get_checkfile(args.checkpoint)
    greedy.load_model(path=checkfile)

    sampler = SamplingDecoder(use_tokenizer=args.use_tokenizer)
    sampler.load_model(path=checkfile)

    while True:
        sentence = input('Input a sentence: ')
        print('Greedy decoder:')
        tree, logprob, *rest = greedy(sentence)
        print('  {} {:.2f}'.format(tree.linearize(with_tag=False), logprob))
        print()

        print('Sampling decoder:')
        for _ in range(10):
            tree, logprob, *rest = sampler(sentence)
            print('  {} {:.2f}'.format(tree.linearize(with_tag=False), logprob))
        print('-'*79)
        print()


def predict_input_gen(args):
    print('Predicting with generative model.')
    assert os.path.exists(args.proposal_model), 'specify valid proposal model.'

    num_samples = 100
    decoder = GenerativeImportanceDecoder(use_tokenizer=True, num_samples=num_samples)
    decoder.load_model(path=args.checkpoint)
    decoder.load_proposal_model(path=args.proposal_model)

    while True:
        sentence = input('Input a sentence: ')

        print('Perplexity: {:.2f}'.format(decoder.perplexity(sentence)))

        print('MAP tree:')
        tree, proposal_logprob, logprob = decoder.map_tree(sentence)
        print('  {} {:.2f} {:.2f}'.format(tree.linearize(with_tag=False), logprob, proposal_logprob))
        print()

        scored = decoder.scored_samples(sentence)
        scored = remove_duplicates(scored)  # For printing purposes.
        print(f'Unique samples: {len(scored)}/{num_samples}.')
        print('Highest q(y|x):')
        scored = sorted(scored, reverse=True, key=lambda t: t[1])
        for tree, proposal_logprob, logprob in scored[:4]:
            print('  {} {:.2f} {:.2f}'.format(tree.linearize(with_tag=False), logprob, proposal_logprob))
        print('Highest p(x,y):')
        scored = sorted(scored, reverse=True, key=lambda t: t[-1])
        for tree, proposal_logprob, logprob in scored[:4]:
            print('  {} {:.2f} {:.2f}'.format(tree.linearize(with_tag=False), logprob, proposal_logprob))
        print('-'*79)
        print()


def sample_generative(args):
    print('Sampling from the generative model.')

    decoder = GenerativeSamplingDecoder()
    decoder.load_model(path=args.checkpoint)

    print('Samples:')
    for i in range(5):
        tree, logprob, _ = decoder()
        print('>', tree.linearize(with_tag=False))
        print()


def sample_proposals(args):
    assert os.path.exists(args.data), 'specifiy file to parse with --data.'

    print(f'Sampling proposal trees for lines in `{args.data}`.')
    with open(args.data, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    if is_tree(lines[0]):
        lines = [Tree.fromstring(line).leaves() for line in lines]

    checkfile = get_checkfile(args.checkpoint)
    decoder = SamplingDecoder(use_tokenizer=False)
    decoder.load_model(path=checkfile)

    num_procs = mp.cpu_count() if args.num_procs == -1 else args.num_procs
    if num_procs == 1:
        samples = []
        for i, line in enumerate(tqdm(lines)):
            for _ in range(args.num_samples):
                tree, logprob, _ = decoder(line, alpha=args.alpha)  # sample a tree
                samples.append(
                    ' ||| '.join((str(i), str(logprob.item()), tree.linearize(with_tag=False))))
    else:
        print(f'Sampling proposals with {num_procs} processors...')
        samples = []
        for i, line in enumerate(tqdm(lines)):
            tuples = decoder.sample_parallel(line, args.num_samples, with_tag=False)
            for tree, logprob in tuples:
                samples.append(
                    ' ||| '.join((str(i), str(logprob), tree)))
    # Write samples.
    with open(args.out, 'w') as f:
        print('\n'.join(samples), file=f, end='')


def main(args):
    if args.from_input:
        if args.rnng_type == 'disc':
            predict_input_disc(args)
        elif args.rnng_type == 'gen':
            predict_input_gen(args)
    elif args.sample_proposals:
        assert args.rnng_type == 'disc'
        sample_proposals(args)
    elif args.from_file:
        predict_file(args)
    elif args.sample_gen:
        assert args.rnng_type == 'gen'
        sample_generative(args)
    else:
        exit('Specify type of prediction. Use --from-input, --from-file or --sample-gen.')
