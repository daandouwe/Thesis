import os
import glob
import tempfile

import dynet as dy
from tqdm import tqdm
import numpy as np

from rnng.decoder import GenerativeDecoder
from utils.trees import fromstring, InternalNode
from utils.evalb import evalb
from utils.general import ceil_div, load_model, is_tree


def predict_text_file(args):
    assert os.path.exists(args.infile), 'specifiy file to parse with --infile.'

    print(f'Predicting trees for lines in `{args.infile}`.')

    with open(args.infile, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    checkfile = get_checkfile(args.checkpoint)

    if args.model_type == 'disc':
        print('Predicting with discriminative model.')
        decoder = GreedyDecoder(use_tokenizer=False)
        decoder.load_model(path=checkfile)

    elif args.model_type == 'gen':
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


def predict_tree_file(args):
    assert os.path.exists(args.infile), 'specifiy file to parse with --infile.'

    print(f'Predicting trees for lines in `{args.infile}`.')

    with open(args.infile, 'r') as f:
        lines = [fromstring(line.strip()).words() for line in f if line.strip()]

    if args.model_type == 'disc':
        print('Loading discriminative model...')
        parser = load_model(args.checkpoint)
        parser.eval()
        print('Done.')

    elif args.model_type == 'gen':
        exit('Not yet...')

        print('Loading generative model...')
        parser = GenerativeDecoder()
        parser.load_model(path=args.checkpoint)
        if args.proposal_model:
            parser.load_proposal_model(path=args.proposal_model)
        if args.proposal_samples:
            parser.load_proposal_samples(path=args.proposal_samples)

    trees = []
    for line in tqdm(lines):
        tree, _ = parser.parse(line)
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


def predict_input_disc(args):
    print('Predicting with discriminative model.')

    parser = load_model(args.checkpoint)

    while True:
        sentence = input('Input a sentence: ')
        words = sentence.split()

        print('Greedy decoder:')
        tree, logprob = parser.parse(words)
        print('  {} {:.2f}'.format(
            tree.linearize(with_tag=False), logprob.value()))
        print()

        print('Sampling decoder:')
        for _ in range(5):
            tree, logprob, *rest = parser.sample(words)
            print('  {} {:.2f}'.format(
                tree.linearize(with_tag=False), logprob.value()))
        print('-'*79)
        print()


def predict_input_gen(args):
    print('Predicting with generative model.')

    joint = load_model(args.checkpoint)
    proposal = load_model(args.proposal_model)
    parser = GenerativeDecoder(
        model=joint, proposal=proposal, num_samples=args.num_samples)

    while True:
        sentence = input('Input a sentence: ')
        words = sentence.split()

        print('Processed:', ' '.join(joint.word_vocab.process(words)))

        print('Perplexity: {:.2f}'.format(parser.perplexity(words)))

        print('MAP tree:')
        tree, proposal_logprob, joint_logprob = parser.map_tree(words)
        print('  {} {:.2f} {:.2f}'.format(
            tree.linearize(with_tag=False), joint_logprob, proposal_logprob))
        print()

        scored = parser.scored_samples(words)

        print(f'Unique samples: {len(scored)}/{args.num_samples}.')

        print('Highest q(y|x):')
        scored = sorted(scored, reverse=True, key=lambda t: t[1])
        for tree, proposal_logprob, joint_logprob, count in scored[:4]:
            print('  {} {:.2f} {:.2f} {}'.format(
                tree.linearize(with_tag=False), joint_logprob, proposal_logprob, count))

        print('Highest p(x,y):')
        scored = sorted(scored, reverse=True, key=lambda t: t[2])
        for tree, proposal_logprob, joint_logprob, count in scored[:4]:
            print('  {} {:.2f} {:.2f} {}'.format(
                tree.linearize(with_tag=False), joint_logprob, proposal_logprob, count))
        print('-'*79)
        print()


def sample_generative(args):
    print('Sampling from the generative model.')

    parser = load_model(args.checkpoint)

    print('Samples:')
    for i in range(args.num_samples):
        tree, logprob = parser.sample()
        print('>', tree.linearize(with_tag=False))
        print()


def predict_perplexity(args):
    assert os.path.exists(args.infile), 'specifiy file to parse with --infile.'

    print(f'Predicting perplexity for lines in `{args.infile}`...')

    with open(args.infile, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    if is_tree(lines[0]):
        sentences = [fromstring(line.strip()).words() for line in lines]
    else:
        sentences = [line.strip().split() for line in lines]

    joint = load_model(args.checkpoint)
    decoder = GenerativeDecoder(
        model=joint, num_samples=args.num_samples)
    decoder.load_proposal_samples(
        args.proposal_samples)

    nlls = []
    lengths = []
    perplexities = []
    processed = []
    for words in tqdm(sentences):
        dy.renew_cg()
        logprob = decoder.logprob(words)
        perplexity = np.exp(-logprob / len(words))
        unked_words = joint.word_vocab.process(words)

        nlls.append(-logprob)
        lengths.append(len(words))
        perplexities.append(perplexity)
        processed.append(unked_words)

    average_perplexity = np.exp(np.sum(nlls) / np.sum(lengths))

    fields = zip(range(len(sentences)), perplexities, nlls, lengths, processed)

    outfile = args.outfile + '.tsv' if not args.outfile.endswith('.tsv') else args.outfile
    print(f'Writing results to `{outfile}`...')

    with open(outfile, 'w') as f:
        print('\t'.join(
                'id', 'perplexity', 'nll', 'length', 'avg-perplexity', 'processed-sentence'),
            file=f)

        for i, pp, nll, length, words in fields:
            results = '\t'.join((
                i,
                pp,
                nll,
                length,
                average_perplexity,
                ' '.join(words)
            ))
            print(results, file=f)

    print('Average perplexity:', round(average_perplexity, 2))


def sample_proposals(args):
    assert os.path.exists(args.infile), 'specifiy file to parse with --infile.'

    print(f'Sampling proposal trees for sentences in `{args.infile}`.')

    with open(args.infile, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    if is_tree(lines[0]):
        sentences = [fromstring(line).words() for line in lines]
    else:
        sentences = [line.split() for line in lines]

    if args.max_lines > 0:
        sentences = sentences[:args.max_lines]

    parser = load_model(args.checkpoint)

    samples = []
    if args.model_type == 'crf':
        for i, words in enumerate(tqdm(sentences)):
            dy.renew_cg()
            for tree, nll in parser.sample(words, num_samples=args.num_samples):
                samples.append(
                    ' ||| '.join(
                        (str(i), str(-nll.value()), tree.linearize(with_tag=False))))
    else:
        for i, words in enumerate(tqdm(sentences)):
            for _ in range(args.num_samples):
                tree, nll = parser.sample(words, alpha=args.alpha)
                samples.append(
                    ' ||| '.join(
                        (str(i), str(-nll.value()), tree.linearize(with_tag=False))))

    with open(args.outfile, 'w') as f:
        print('\n'.join(samples), file=f, end='')


def inspect_model(args):
    print(f'Inspecting attention for sentences in `{args.infile}`.')

    parser = load_model(args.checkpoint)

    with open(args.infile, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    lines = lines[:args.max_lines]
    if is_tree(lines[0]):
        sentences = [fromstring(line).words() for line in lines]
    else:
        sentences = [line.split() for line in lines]

    def inspect_after_reduce(parser):
        subtree = parser.stack._stack[-1].subtree
        head = subtree.label
        children = [child.label
            if isinstance(child, InternalNode) else child.word
            for child in subtree.children]
        attention = parser.composer._attn
        gate = np.mean(parser.composer._gate)
        attention = [attention] if not isinstance(attention, list) else attention  # in case .value() returns a float
        attentive = [f'{child} ({attn:.2f})'
            for child, attn in zip(children, attention)]
        print('  ', head, '|', ' '.join(attentive), f'[{gate:.2f}]')

    def parse_with_inspection(parser, words):
        parser.eval()
        nll = 0.
        word_ids = [parser.word_vocab.index_or_unk(word) for word in words]
        parser.initialize(word_ids)
        while not parser.stack.is_finished():
            u = parser.parser_representation()
            action_logits = parser.f_action(u)
            action_id = np.argmax(action_logits.value() + parser._add_actions_mask())
            nll += dy.pickneglogsoftmax(action_logits, action_id)
            parser.parse_step(action_id)
            if action_id == parser.REDUCE_ID:
                inspect_after_reduce(parser)
        tree = parser.get_tree()
        tree.substitute_leaves(iter(words))  # replaces UNKs with originals
        return tree, nll

    for sentence in sentences:
        tree, _ = parser.parse(sentence)
        print('>', ' '.join(sentence))
        print('>', tree.linearize(with_tag=False))
        parse_with_inspection(parser, sentence)
        print()


def main(args):
    if args.from_input:
        if args.model_type in ('disc-rnng', 'crf'):
            predict_input_disc(args)
        elif args.model_type == 'gen-rnng':
            predict_input_gen(args)
    elif args.from_text_file:
        predict_text_file(args)
    elif args.from_tree_file:
        predict_tree_file(args)
    elif args.perplexity:
        assert args.model_type == 'gen-rnng'
        predict_perplexity(args)
    elif args.sample_proposals:
        assert args.model_type in ('disc-rnng', 'crf')
        sample_proposals(args)
    elif args.sample_gen:
        assert args.model_type == 'gen-rrng'
        sample_generative(args)
    elif args.inspect_model:
        assert args.model_type.endswith('rnng')
        inspect_model(args)
    else:
        exit('Specify type of prediction. Use --from-input, --from-file or --sample-gen.')
