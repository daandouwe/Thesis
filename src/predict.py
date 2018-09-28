#!/usr/bin/env python
import os
import glob

from decode import GreedyDecoder, BeamSearchDecoder, SamplingDecoder


def predict(args):
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
        print('{} {:.2f}'.format(tree.linearize(with_tag=False), logprob))
        print()

        print('Sampling decoder:')
        for _ in range(3):
            tree, logprob, *rest = sampler(sentence)
            print('{} {:.2f}'.format(tree.linearize(with_tag=False), logprob))
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
