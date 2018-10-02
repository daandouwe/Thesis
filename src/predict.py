#!/usr/bin/env python
import os
import glob

from decode import GreedyDecoder, BeamSearchDecoder, SamplingDecoder, GenerativeDecoder


def predict(args):
    if args.model == 'disc':
        predict_disc(args)
    elif args.model == 'gen':
        predict_gen(args)


def remove_duplicates(samples):
    output = []
    seen = set()
    for tree, proposal_logprob, logprob in samples:
        if tree.linearize() not in seen:
            output.append((tree, proposal_logprob, logprob))
            seen.add(tree.linearize())
    return output


def predict_gen(args):
    print('Predicting with generative model.')
    disc_checkfile = args.proposal
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

        scored = decoder.rank(sentence)
        scored = remove_duplicates(scored)  # For printing purposes.
        print(f'Number of unique samples: {len(scored)}.')
        print('Best p(x,y):')
        scored = sorted(scored, reverse=True, key=lambda t: t[-1])
        for tree, proposal_logprob, logprob in scored[:10]:
            print('  {} {:.2f} {:.2f}'.format(tree.linearize(with_tag=False), logprob, proposal_logprob))
        print('Best q(y|x):')
        scored = sorted(scored, reverse=True, key=lambda t: t[1])
        for tree, proposal_logprob, logprob in scored[:10]:
            print('  {} {:.2f} {:.2f}'.format(tree.linearize(with_tag=False), logprob, proposal_logprob))
        print()


def predict_disc(args):
    print('Predicting with discriminative model.')
    if not args.checkpoint:
        latest_dir = max(glob.glob(os.path.join('checkpoints', '*/')))
        checkfile = os.path.join(latest_dir, 'model.pt')
    else:
        checkfile = args.checkpoint

    greedy = GreedyDecoder(use_tokenizer=args.use_tokenizer)
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
