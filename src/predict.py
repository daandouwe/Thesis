import os
import glob
import tempfile

import dynet as dy
from tqdm import tqdm
import numpy as np

from rnng.decoder import GenerativeDecoder
from rnng.model import DiscRNNG
from utils.trees import fromstring, InternalNode
from utils.evalb import evalb
from utils.general import ceil_div, load_model, is_tree


def predict_text_file(args):
    assert os.path.exists(args.infile), 'specifiy file to parse with --infile.'

    print(f'Predicting trees for lines in `{args.infile}`.')

    with open(args.infile, 'r') as f:
        lines = [line.strip().split() for line in f.readlines()]

    if args.model_type == 'disc-rnng':
        print('Predicting with discriminative RNNG.')
        parser = load_model(args.checkpoint)
    elif args.model_type == 'crf':
        print('Predicting with CRF parser.')
        parser = load_model(args.checkpoint)
    elif args.model_type == 'gen-rnng':
        print('Predicting with generative RNNG.')
        model = load_model(args.checkpoint)
        parser = GenerativeDecoder(model=model)
        if args.proposal_model:
            parser.load_proposal_model(args.proposal_model)
        elif args.proposal_samples:
            parser.load_proposal_samples(args.proposal_samples)
        else:
            raise ValueError('Specify proposals.')
    else:
        raise ValueError('Specify model-type.')

    print(f'Predicting trees for `{args.infile}`...')
    trees = []
    for words in tqdm(lines):
        dy.renew_cg()
        tree, *rest = parser.parse(words)
        trees.append(tree.linearize(with_tag=True))
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
    print('Predicting with discriminative rnng.')

    parser = load_model(args.checkpoint)

    while True:
        sentence = input('Input a sentence: ')
        words = sentence.split()

        print('Processed:', ' '.join(parser.word_vocab.process(words)))
        print()

        print('Parse:')
        tree, nll = parser.parse(words)
        print('  {} {:.2f}'.format(
            tree.linearize(with_tag=False), nll.value()))
        print()

        print('Samples (alpha = {}):'.format(args.alpha, 2))
        for _ in range(8):
            tree, nll, *rest = parser.sample(words, alpha=args.alpha)
            print('  {} {:.2f}'.format(
                tree.linearize(with_tag=False), nll.value()))
        print('-'*79)
        print()


def predict_input_crf(args):
    print('Predicting with crf parser.')

    parser = load_model(args.checkpoint)

    ##
    right_branching = fromstring("(S (NP (@ The) (@ (@ other) (@ (@ hungry) (@ cat)))) (@ (VP meows ) (@ .)))").convert()
    left_branching = fromstring("(S (NP (@ The) (@ (@ (@ other) (@ hungry)) (@ cat))) (@ (VP meows ) (@ .)))").convert()

    # right_branching = fromstring("(X (X (@ The) (@ (@ other) (@ (@ hungry) (@ cat)))) (@ (X meows ) (@ .)))").convert()
    # left_branching = fromstring("(X (X (@ The) (@ (@ (@ other) (@ hungry)) (@ cat))) (@ (X meows ) (@ .)))").convert()

    right_nll = parser.forward(right_branching, is_train=False)
    left_nll = parser.forward(left_branching, is_train=False)

    print('Right:', right_nll.value())
    print('Left:', left_nll.value())
    ##

    while True:
        sentence = input('Input a sentence: ')
        words = sentence.split()

        print('Processed:', ' '.join(parser.word_vocab.process(words)))
        print()

        print('Parse:')
        tree, nll = parser.parse(words)
        print('  {} {:.3f}'.format(
            tree.linearize(with_tag=False), nll.value()))
        print()

        print('Samples:')
        parse, parse_logprob, samples, entropy = parser.parse_sample_entropy(
            words, num_samples=8, alpha=1)
        for tree, nll in samples:
            print('  {} {:.3f}'.format(
                tree.linearize(with_tag=False), nll.value()))
            # print('  {} ||| {} ||| {:.3f}'.format(
            #     tree.convert().linearize(with_tag=False), tree.un_cnf().linearize(with_tag=False), nll.value()))
            print()
        print('Parse (temperature {}):'.format(args.alpha))
        print('  {} {:.3f}'.format(
            parse.linearize(with_tag=False), -parse_logprob.value()))
        print()

        print('Entropy:')
        print('  {:.3f}'.format(entropy.value()))

        print('-'*79)
        print()


def predict_input_gen(args):
    print('Predicting with generative rnng.')

    joint = load_model(args.checkpoint)
    proposal = load_model(args.proposal_model)

    parser = GenerativeDecoder(
        model=joint, proposal=proposal, num_samples=args.num_samples, alpha=args.alpha)

    while True:
        sentence = input('Input a sentence: ')
        words = sentence.split()

        print('Processed:', ' '.join(joint.word_vocab.process(words)))
        print()

        print('Perplexity: {:.2f}'.format(parser.perplexity(words)))
        print()

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

    np.random.seed(args.numpy_seed)

    with open(args.infile, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    if is_tree(lines[0]):
        sentences = [fromstring(line.strip()).words() for line in lines]
    else:
        sentences = [line.strip().split() for line in lines]

    model = load_model(args.checkpoint)
    proposal = load_model(args.proposal_model)
    decoder = GenerativeDecoder(
        model=model, proposal=proposal, num_samples=args.num_samples, alpha=args.alpha)

    proposal_type = 'disc-rnng' if isinstance(proposal, DiscRNNG) else 'crf'

    filename_base = 'proposal={}_num-samples={}_temp={}_seed={}'.format(
        proposal_type, args.num_samples, args.alpha, args.numpy_seed)
    proposals_path = os.path.join(args.outdir, filename_base + '.props')
    result_path = os.path.join(args.outdir, filename_base + '.tsv')

    print('Predicting perplexity with Generative RNNG.')
    print(f'Loading model from `{args.checkpoint}`.')
    print(f'Loading proposal from `{args.proposal_model}`.')
    print(f'Loading lines from directory `{args.infile}`.')
    print(f'Writing proposals to `{proposals_path}`.')
    print(f'Writing predictions to `{result_path}`.')

    print('Sampling proposals...')
    decoder.generate_proposal_samples(sentences, proposals_path)
    print('Computing perplexity...')
    _, perplexity = decoder.predict_from_proposal_samples(proposals_path)

    with open(result_path, 'w') as f:
        print('\t'.join((
                'proposal',
                'file',
                'perplexity',
                'num-samples',
                'temp',
                'seed'
            )),
            file=f)
        print('\t'.join((
                proposal_type,
                os.path.basename(args.infile),
                str(perplexity),
                str(args.num_samples),
                str(args.alpha),
                str(args.numpy_seed)
            )),
            file=f)


def predict_perplexity_from_samples(args):

    print('Predicting perplexity with Generative RNNG.')
    print(f'Loading model from `{args.checkpoint}`.')
    print(f'Loading proposal samples from `{args.proposal_samples}`.')
    print(f'Loading lines from directory `{args.infile}`.')
    print(f'Writing predictions to `{args.outfile}`.')

    np.random.seed(args.numpy_seed)

    model = load_model(args.checkpoint)
    decoder = GenerativeDecoder(
        model=model, num_samples=args.num_samples, alpha=args.alpha)

    print('Computing perplexity...')
    trees, perplexity = decoder.predict_from_proposal_samples(args.proposal_samples)

    # Compute f-score from trees
    base_name = os.path.splitext(args.outfile)[0]
    pred_path = base_name + '.trees'
    result_path = base_name + '.result'
    with open(pred_path, 'w') as f:
        print('\n'.join(trees), file=f)
    fscore = evalb(
        args.evalb_dir, pred_path, args.infile, result_path)

    print(f'Results: {fscore} fscore, {perplexity} perplexity.')

    with open(args.outfile, 'w') as f:
        print(
            'proposals',
            'perplexity',
            'fscore',
            'num-samples',
            'temp',
            'seed',
            sep='\t', file=f)
        print(
            args.proposal_samples,
            perplexity,
            fscore,
            args.num_samples,
            args.alpha,
            args.numpy_seed,
            sep='\t', file=f
        )


def sample_proposals(args):
    assert os.path.exists(args.infile), 'specifiy file to parse with --infile.'

    print(f'Sampling proposal trees for sentences in `{args.infile}`.')

    with open(args.infile, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    if is_tree(lines[0]):
        sentences = [fromstring(line).words() for line in lines]
    else:
        sentences = [line.split() for line in lines]

    parser = load_model(args.checkpoint)

    samples = []
    if args.model_type == 'crf':
        for i, words in enumerate(tqdm(sentences)):
            dy.renew_cg()
            for tree, nll in parser.sample(words, num_samples=args.num_samples):
                samples.append(
                    ' ||| '.join(
                        (str(i), str(-nll.value()), tree.linearize(with_tag=False))))
                print(
                    ' ||| '.join(
                        (str(i), str(-nll.value()), tree.linearize(with_tag=False))))

    else:
        for i, words in enumerate(tqdm(sentences)):
            for _ in range(args.num_samples):
                dy.renew_cg()
                tree, nll = parser.sample(words, alpha=args.alpha)
                samples.append(
                    ' ||| '.join(
                        (str(i), str(-nll.value()), tree.linearize(with_tag=False))))

    with open(args.outfile, 'w') as f:
        print('\n'.join(samples), file=f, end='')


def inspect_model(args):
    assert args.model_type == 'disc-rnng', args.model_type

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


def predict_entropy(args):
    print(f'Predicting entropy for lines in `{args.infile}`, writing to `{args.outfile}`...')
    print(f'Loading model from `{args.checkpoint}`.')
    print(f'Using {args.num_samples} samples.')

    parser = load_model(args.checkpoint)
    parser.eval()

    with open(args.infile, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    if is_tree(lines[0]):
        sentences = [fromstring(line.strip()).words() for line in lines]
    else:
        sentences = [line.strip().split() for line in lines]

    with open(args.outfile, 'w') as f:
        print('id', 'entropy', 'num-samples', 'model', 'file', file=f, sep='\t')
        for i, words in enumerate(tqdm(sentences)):
            dy.renew_cg()
            if args.num_samples == 0:
                assert args.model_type == 'crf', 'exact computation only for crf.'
                entropy = parser.entropy(words)
            else:
                if args.model_type == 'crf':
                    samples = parser.sample(words, num_samples=args.num_samples)
                    if args.num_samples == 1:
                        samples = [samples]
                else:
                    samples = [parser.sample(words, alpha=args.alpha) for _ in range(args.num_samples)]
                trees, nlls = zip(*samples)
                entropy = dy.esum(list(nlls)) / len(nlls)
            print(i, entropy.value(), args.num_samples, args.model_type, args.infile, file=f, sep='\t')

def main(args):
    if args.from_input:
        if args.model_type == 'disc-rnng':
            predict_input_disc(args)
        elif args.model_type == 'crf':
            predict_input_crf(args)
        elif args.model_type == 'gen-rnng':
            predict_input_gen(args)
        else:
            exit('Invalid option for prediction: {}'.format(args.model_type))
    elif args.from_text_file:
        predict_text_file(args)
    elif args.from_tree_file:
        predict_tree_file(args)
    elif args.perplexity:
        assert args.model_type == 'gen-rnng', args.model_type
        if args.proposal_model:
            predict_perplexity(args)
        else:
            predict_perplexity_from_samples(args)
    elif args.sample_proposals:
        assert args.model_type in ('disc-rnng', 'crf'), args.model_type
        sample_proposals(args)
    elif args.sample_gen:
        assert args.model_type == 'gen-rnng', args.model_type
        sample_generative(args)
    elif args.inspect_model:
        assert args.model_type.endswith('rnng'), args.model_type
        inspect_model(args)
    elif args.entropy:
        assert args.model_type in ('disc-rnng', 'crf'), args.model_type
        predict_entropy(args)
    else:
        exit('Specify type of prediction. Use --from-input, --from-file or --sample-gen.')
