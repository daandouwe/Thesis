#!/usr/bin/env python
import glob
import os

import torch
from tensorboardX import SummaryWriter

from data import Item, Action, Corpus
from scripts.get_oracle import unkify

def process_unks(sent, dictionary):
    new_sent = []
    for word in sent:
        try:
            dictionary[word]
            new_sent.append(word)
        except KeyError:
            unk = unkify([word], dictionary)[0]
            new_sent.append(unk)
    return new_sent

def predict(model, sent, dictionary, writer=None):
    model.eval()
    indices = [Item(word, dictionary[word]) for word in sent]
    if writer is not None:
        return parse_and_write_embeddings(model, indices, writer)
    else:
        return model.parse(indices)

def parse_and_write_embeddings(model, sentence, writer):
    model.initialize(sentence)
    t = 0
    while not model.stack.empty:
        top_token = model.stack.top_item.token
        embedding = model.stack.top_item.embedding
        encoding = model.stack.top_item.encoding
        writer.add_embedding(embedding, metadata=[top_token], global_step=t, tag='Embedding')
        writer.add_embedding(encoding, metadata=[top_token], global_step=t, tag='Encoding')

        t += 1
        # Compute loss
        stack, buffer, history = model.get_encoded_input()
        x = torch.cat((buffer, history, stack), dim=-1)
        action_logits = model.action_mlp(x)
        # Get highest scoring valid predictions.
        vals, ids = action_logits.sort(descending=True)
        vals, ids = vals.data.squeeze(0), ids.data.squeeze(0)
        i = 0
        action = Action(model.dictionary.i2a[ids[i]], ids[i])
        while not model.is_valid_action(action):
            i += 1
            action = Action(model.dictionary.i2a[ids[i]], ids[i])
        if action.index == model.OPEN:
            nonterminal_logits = model.nonterminal_mlp(x)
            vals, ids = nonterminal_logits.sort(descending=True)
            vals, ids = vals.data.squeeze(0), ids.data.squeeze(0)
            action.symbol = Item(model.dictionary.i2n[ids[0]], ids[0], nonterminal=True)
        model.parse_step(action)
    return model.stack.tree.linearize()

def main(args):
    # Set cuda.
    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    print('Device: {}.'.format(args.device))

    print('Loading corpus with textline {}.'.format(args.textline))
    corpus = Corpus(data_path=args.data, textline=args.textline, char=args.use_char)

    latest_dir = max(glob.glob(os.path.join('checkpoints', '*/')))
    checkfile = os.path.join(latest_dir, 'model.pt')
    print('Loading model from {}.'.format(checkfile))
    # Load best saved model.
    with open(checkfile, 'rb') as f:
        model = torch.load(f)
    model.to(args.device)

    if writer:
        print(f'Created tensorboard summary writer at {args.logdir}.')
        writer = SummaryWriter(latest_dir)

    while True:
        sent = input('Input a sentence: ')
        words = sent.split()
        unked = process_unks(words, corpus.dictionary.w2i)
        tree = predict(model, unked, corpus.dictionary.w2i, writer=writer)
        print(words)
        print(unked)
        print(tree)
        print()

if __name__ == '__main__':
    main(args)
