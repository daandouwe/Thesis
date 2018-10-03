#!/usr/bin/env python
import os
import glob
import tempfile

from tqdm import tqdm
from nltk import Tree

from decode import GreedyDecoder, BeamSearchDecoder, SamplingDecoder, GenerativeDecoder
from eval import evalb


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
    """Simple `oracle` to know if line is a tree."""
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
    if args.model == 'disc':
        print('Predicting with discriminative model.')
        decoder = GreedyDecoder(use_tokenizer=False)
        decoder.load_model(path=checkfile)
    elif args.model == 'gen':
        print('Predicting with generative model.')
        decoder = GenerativeDecoder(use_tokenizer=False)
        decoder.load_model(path=checkfile)
        if args.proposal_model:
            decoder.load_proposal_model(path=args.proposal_model)
        if args.proposal_samples:
            decoder.load_proposal_samples(path=args.proposal_samples)

    print(f'Predicting trees for `{args.data}`...')
    trees = []
    for line in tqdm(lines):
        tree, *rest = decoder(line)
        trees.append(tree)

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
        for _ in range(3):
            tree, logprob, *rest = sampler(sentence)
            print('  {} {:.2f}'.format(tree.linearize(with_tag=False), logprob))
        print('-'*79)
        print()


def predict_input_gen(args):
    print('Predicting with generative model.')
    disc_checkfile = args.proposal_model
    gen_checkfile = args.checkpoint

    decoder = GenerativeDecoder(use_tokenizer=args.use_tokenizer)
    decoder.load_models(gen_path=gen_checkfile, disc_path=disc_checkfile)

    while True:
        sentence = input('Input a sentence: ')

        prob = decoder.prob(sentence)
        nll = prob.log() / len(sentence)
        perplexity = nll.exp().item()
        print('Perplexity {:.2f}'.format(perplexity))

        print('Tree:')
        tree, proposal_logprob, logprob = decoder.map_tree(sentence)
        print('  {} {:.2f} {:.2f}'.format(tree.linearize(with_tag=False), logprob, proposal_logprob))
        print()

        scored = decoder.scored_samples(sentence)
        scored = remove_duplicates(scored)  # For printing purposes.
        print(f'Number of unique samples: {len(scored)}.')
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


def main(args):
    if args.from_input:
        if args.model == 'disc':
            predict_input_disc(args)
        elif args.model == 'gen':
            predict_input_gen(args)
    elif args.from_file:
        predict_file(args)


if __name__ == '__main__':
    # TODO: Log embeddings while predicting:
    if writer:
        print(f'Created tensorboard summary writer at {args.logdir}.')
        writer = SummaryWriter(latest_dir)

    tree = model.stack.tree.linearize() # partial tree
    top_token = model.stack.top_item.token
    embedding = model.stack.top_item.embedding
    encoding = model.stack.top_item.encoding
    writer.add_text('Tree', metadata=[top_token], global_step=t, tag='Encoding')
    writer.add_embedding(embedding, metadata=[top_token], global_step=t, tag='Embedding')
    writer.add_embedding(encoding, metadata=[top_token], global_step=t, tag='Encoding')
