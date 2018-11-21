#!/usr/bin/env python
import os
import glob
import tempfile

import dynet as dy
from tqdm import tqdm
from nltk import Tree
import numpy as np

from decode import GenerativeDecoder
from tree import fromstring, InternalNode
from eval import evalb
from utils import ceil_div


def load_model(dir):
    model = dy.ParameterCollection()
    [rnng] = dy.load(os.path.join(dir, 'model'), model)
    return rnng


def is_tree(line):
    """Simple `oracle` to see if line is a tree."""
    assert isinstance(line, str), line
    try:
        Tree.fromstring(line)
        return True
    except ValueError:
        return False


def predict_tree_file(args):
    assert os.path.exists(args.infile), 'specifiy file to parse with --infile.'

    print(f'Predicting trees for lines in `{args.infile}`.')

    with open(args.infile, 'r') as f:
        lines = [fromstring(line.strip()).leaves() for line in f if line.strip()]

    if args.rnng_type == 'disc':
        print('Loading discriminative model...')
        decoder = load_model(args.checkpoint)
        decoder.eval()
        print('Done.')

    elif args.rnng_type == 'gen':
        exit('Not yet...')

        print('Loading generative model...')
        decoder = GenerativeDecoder()
        decoder.load_model(path=args.checkpoint)
        if args.proposal_model:
            decoder.load_proposal_model(path=args.proposal_model)
        if args.proposal_samples:
            decoder.load_proposal_samples(path=args.proposal_samples)

    trees = []
    for line in tqdm(lines):
        tree, _ = decoder.parse(line)
        trees.append(tree.linearize())

    pred_path = os.path.join(args.outfile)
    result_path = args.outfile + '.results'
    # Save the predicted trees.
    with open(pred_path, 'w') as f:
        print('\n'.join(trees), file=f)
    # Score the trees.
    fscore = evalb(args.evalb_dir, pred_path, args.infile, result_path)
    print(f'Predictions saved in `{pred_path}`. Results saved in `{result_path}`.')
    print(f'F-score {fscore:.2f}.')


def predict_text_file(args):
    assert os.path.exists(args.infile), 'specifiy file to parse with --infile.'
    print(f'Predicting trees for lines in `{args.infile}`.')
    with open(args.infile, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    checkfile = get_checkfile(args.checkpoint)

    if args.rnng_type == 'disc':
        print('Predicting with discriminative model.')
        decoder = GreedyDecoder(use_tokenizer=False)
        decoder.load_model(path=checkfile)

    elif args.rnng_type == 'gen':
        print('Predicting with generative model.')
        decoder = GenerativeDecoder(use_tokenizer=False)
        decoder.load_model(path=checkfile)
        if args.proposal_model:
            decoder.load_proposal_model(path=args.proposal_model)
        if args.proposal_samples:
            decoder.load_proposal_samples(path=args.proposal_samples)

    print(f'Predicting trees for `{args.infile}`...')
    trees = []
    for line in tqdm(lines):
        tree, *rest = decoder(line)
        trees.append(tree.linearize(with_tag=False))
    print(f'Saved predicted trees in `{args.outfile}`.')
    with open(args.outfile, 'w') as f:
        print('\n'.join(trees), file=f)


def predict_from_tree(gold_tree):
    """Predicts from a gold tree input and computes fscore with prediction.

    Input should be a unicode string in the :
        u'(S (NP (DT The) (NN equity) (NN market)) (VP (VBD was) (ADJP (JJ illiquid))) (. .))'
    """
    # Make a temporay directory for the EVALB files.
    evalb_dir = os.path.expanduser('~/EVALB')  # TODO: this should be part of args.
    temp_dir = tempfile.TemporaryDirectory(prefix='evalb-')
    gold_path = os.path.join(temp_dir.name, 'gold.txt')
    pred_path = os.path.join(temp_dir.name, 'predicted.txt')
    result_path = os.path.join(temp_dir.name, 'output.txt')

    # Extract sentence from the gold tree.
    sent = Tree.fromstring(gold_tree).leaves()

    # Predict a tree for the sentence.
    pred_tree, *rest = self(sent)
    pred_tree = pred_tree.linearize()
    # Dump these in the temp-file.
    with open(gold_path, 'w') as f:
        print(gold_tree, file=f)
    with open(pred_path, 'w') as f:
        print(pred_tree, file=f)
    fscore = evalb(evalb_dir, pred_path, gold_path, result_path)

    # Cleanup the temporary directory.
    temp_dir.cleanup()

    return pred_tree, fscore


def predict_input_disc(args):
    print('Predicting with discriminative model.')

    rnng = load_model(args.checkpoint)

    while True:
        sentence = input('Input a sentence: ')
        words = sentence.split()

        print('Greedy decoder:')
        tree, logprob = rnng.parse(words)
        print('  {} {:.2f}'.format(
            tree.linearize(with_tag=False), logprob.value()))
        print()

        print('Sampling decoder:')
        for _ in range(5):
            tree, logprob, *rest = rnng.sample(words)
            print('  {} {:.2f}'.format(
                tree.linearize(with_tag=False), logprob.value()))
        print('-'*79)
        print()


def predict_input_gen(args):
    print('Predicting with generative model.')
    assert os.path.exists(args.proposal_model), 'specify valid proposal model.'

    model = load_model(args.checkpoint)
    proposal = load_model(args.proposal_model)
    decoder = GenerativeDecoder(
        model=model, proposal=proposal, num_samples=args.num_samples)

    while True:
        sentence = input('Input a sentence: ')
        words = sentence.split()

        print('Perplexity: {:.2f}'.format(decoder.perplexity(words)))

        print('MAP tree:')
        tree, proposal_logprob, joint_logprob = decoder.map_tree(words)
        print('  {} {:.2f} {:.2f}'.format(
            tree.linearize(with_tag=False), joint_logprob, proposal_logprob))
        print()

        scored = decoder.scored_samples(words, remove_duplicates=True)

        print(f'Unique samples: {len(scored)}/{args.num_samples}.')

        print('Highest q(y|x):')
        scored = sorted(scored, reverse=True, key=lambda t: t[1])
        for tree, proposal_logprob, joint_logprob in scored[:4]:
            print('  {} {:.2f} {:.2f}'.format(
                tree.linearize(with_tag=False), joint_logprob, proposal_logprob))

        print('Highest p(x,y):')
        scored = sorted(scored, reverse=True, key=lambda t: t[-1])
        for tree, proposal_logprob, joint_logprob in scored[:4]:
            print('  {} {:.2f} {:.2f}'.format(
                tree.linearize(with_tag=False), joint_logprob, proposal_logprob))
        print('-'*79)
        print()


def sample_generative(args):
    print('Sampling from the generative model.')

    model = load_model(args.checkpoint)

    print('Samples:')
    for i in range(args.num_samples):
        tree, logprob = model.sample()
        print('>', tree.linearize(with_tag=False))
        print()


def predict_perplexity(args):
    assert os.path.exists(args.infile), 'specifiy file to parse with --infile.'

    print(f'Predicting perplexity for lines in `{args.infile}`...')
    with open(args.infile, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    if is_tree(lines[0]):
        sentences = [list(fromstring(line.strip()).leaves()) for line in lines]
    else:
        sentences = [line.strip().split() for line in lines]

    model = load_model(args.checkpoint)
    decoder = GenerativeDecoder(model=model, num_samples=args.num_samples)
    decoder.load_proposal_samples(args.proposal_samples)

    pps = []
    for words in tqdm(sentences):
        dy.renew_cg()
        pp = decoder.perplexity(words)
        pps.append(pp)
    avg_pp = np.mean(pps)  # TODO: is this correct?

    with open(args.outfile, 'w') as f:
        print('Average perplexity:', avg_pp)
        for pp, words in zip(pps, sentences):
            print(pp, '|||', ' '.join(words), file=f)


def sample_proposals(args):
    assert os.path.exists(args.infile), 'specifiy file to parse with --infile.'

    print(f'Sampling proposal trees for sentences in `{args.infile}`.')
    with open(args.infile, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    if is_tree(lines[0]):
        sentences = [list(fromstring(line).leaves()) for line in lines]
    else:
        sentences = [line.split() for line in lines]

    parser = load_model(args.checkpoint)

    samples = []
    for i, words in enumerate(tqdm(sentences)):
        dy.renew_cg()
        for _ in range(args.num_samples):
            tree, nll = parser.sample(words, alpha=args.alpha)
            samples.append(
                ' ||| '.join((str(i), str(-nll.value()), tree.linearize(with_tag=False))))

    with open(args.outfile, 'w') as f:
        print('\n'.join(samples), file=f, end='')


def predict_syneval(args):
    assert os.path.exists(args.proposal_model), 'specify valid proposal model.'

    model = load_model(args.checkpoint)
    decoder = GenerativeDecoder(model=model, num_samples=args.num_samples)
    decoder.load_proposal_model(args.proposal_model)

    with open(args.infile + '.pos') as f:
        pos_sents = [line.strip() for line in f.readlines()]

    with open(args.infile + '.neg') as f:
        neg_sents = [line.strip() for line in f.readlines()]

    correct = 0
    with open(args.outfile, 'w') as f:
        for i, (pos, neg) in enumerate(zip(pos_sents, neg_sents)):
            pos_pp = decoder.perplexity(pos)
            neg_pp = decoder.perplexity(neg)
            correct += (pos_pp < neg_pp)
            print(i, round(pos_pp, 2), round(neg_pp, 2), pos_pp < neg_pp, correct, neg)
            print(i, round(pos_pp, 2), round(neg_pp, 2), pos_pp < neg_pp, neg, file=f)

    print(f'Syneval: {correct}/{len(pos_sents)} = {correct / len(pos_sents):%} correct')


def inspect_model(args):
    print(f'Inspecting attention for sentences in `{args.infile}`.')

    model = load_model(args.checkpoint)

    with open(args.infile, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    lines = lines[:args.max_lines]
    if is_tree(lines[0]):
        sentences = [list(fromstring(line).leaves()) for line in lines]
    else:
        sentences = [line.split() for line in lines]

    def inspect_after_reduce(model):
        subtree = model.stack._stack[-1].subtree
        head = subtree.label
        children = [child.label if isinstance(child, InternalNode) else child.word for child in subtree.children]
        attention = model.composer._attn
        gate = np.mean(model.composer._gate)
        attention = [attention] if not isinstance(attention, list) else attention  # in case .value() returns a float
        attentive = [f'{child} ({attn:.2f})'
            for child, attn in zip(children, attention)]
        print('  ', head, '|', ' '.join(attentive), f'[{gate:.2f}]')

    def parse_with_inspection(model, words):
        words = list(words)
        model.eval()
        nll = 0.
        model.initialize(model.word_vocab.indices(words))
        while not model.stack.is_finished():
            u = model.parser_representation()
            action_logits = model.f_action(u)
            action_id = np.argmax(action_logits.value() + model._add_actions_mask())
            nll += dy.pickneglogsoftmax(action_logits, action_id)
            model.parse_step(action_id)
            if action_id == model.REDUCE_ID:
                inspect_after_reduce(model)
        tree = model.get_tree()
        tree.substitute_leaves(iter(words))  # replaces UNKs with originals
        return tree, nll

    for sentence in sentences:
        parse_with_inspection(model, sentence)
        print()


def main(args):
    if args.from_input:
        if args.rnng_type == 'disc':
            predict_input_disc(args)
        elif args.rnng_type == 'gen':
            predict_input_gen(args)
    elif args.from_tree_file:
        predict_tree_file(args)
    elif args.from_text_file:
        predict_text_file(args)
    elif args.perplexity:
        assert args.rnng_type == 'gen'
        predict_perplexity(args)
    elif args.sample_proposals:
        assert args.rnng_type == 'disc'
        sample_proposals(args)
    elif args.sample_gen:
        assert args.rnng_type == 'gen'
        sample_generative(args)
    elif args.inspect_model:
        inspect_model(args)
    elif args.syneval:
        assert args.rnng_type == 'gen'
        predict_syneval(args)
    else:
        exit('Specify type of prediction. Use --from-input, --from-file or --sample-gen.')
