#!/usr/bin/env python
import os
import argparse

import train
import predict_input

def main():

    parser = argparse.ArgumentParser(description='Discriminative RNNG parser',
                                     fromfile_prefix_chars='@') # Enable loading args from textfile
    # Choose mode
    parser.add_argument('mode', type=str, choices=['train', 'predict'])
    # Debugging
    parser.add_argument('-d', '--debug', action='store_true')
    # Data arguments
    parser.add_argument('--data', type=str, default='../tmp',
                        help='location of the data corpus')
    parser.add_argument('--textline', type=str, choices=['unked', 'lower', 'upper'], default='unked',
                        help='textline to use from the oracle file')
    parser.add_argument('--root', type=str, default='.',
                        help='root dir to make output log and checkpoint folders')
    parser.add_argument('--logdir', default=None,
                        help='to be constructed by util.make_folders')
    parser.add_argument('--logfile', default=None,
                        help='to be constructed by util.make_folders')
    parser.add_argument('--disable-folders', action='store_true',
                        help='do not make output folders (debug)')
    # Model arguments
    parser.add_argument('--use-char', action='store_true',
                        help='use character-level word embeddings')
    parser.add_argument('--word-emb-dim', type=int, default=100,
                        help='dim of embeddings for word')
    parser.add_argument('--action-emb-dim', type=int, default=20,
                        help='dim of embeddings for actions')
    parser.add_argument('--word-lstm-hidden', type=int, default=128,
                        help='size of lstm hidden states for StackLSTM and BufferLSTM')
    parser.add_argument('--action-lstm-hidden', type=int, default=128,
                        help='size of lstm hidden states for history encoder')
    parser.add_argument('--lstm-num-layers', type=int, default=2,
                        help='number of layers in lstm')
    parser.add_argument('--mlp-dim', type=int, default=128,
                        help='size of mlp hidden state')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate for embeddings, lstm, and mlp')
    parser.add_argument('--use-glove', action='store_true',
                        help='using pretrained glove embeddings')
    parser.add_argument('--use-fasttext', action='store_true',
                        help='using pretrained fasttext embeddings')
    parser.add_argument('--glove-dir', type=str, default='~/glove',
                        help='to be constructed by util.make_folders')
    parser.add_argument('--glove-torchtext', action='store_true',
                        help='loading glove with torchtext')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed to use')
    # Training arguments
    parser.add_argument('--epochs', default=None,
                        help='max number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='size of mini batch')
    parser.add_argument('--lr', type=float, default=0.0008,
                        help='initial learning rate')
    parser.add_argument('--learning-rate-warmup_steps', type=int,
                        default=160)
    parser.add_argument('--step-decay', type=bool, default=True,
                        help='scheduler parameter')
    parser.add_argument('--step-decay-patience', type=int,default=1,
                        help='scheduler parameter')
    parser.add_argument('--step-decay-factor', type=float,default=0.5,
                        help='scheduler parameter')
    parser.add_argument('--disable-glorot', action='store_true',
                        help='override custom lstm initialization with glorot')
    parser.add_argument('--clip', type=float, default=5.,
                        help='clipping gradient norm at this value')
    parser.add_argument('--print-every', type=int, default=10,
                        help='when to print training progress')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    args = parser.parse_args()

    if args.mode == 'train':
        train.main(args)
    elif args.mode == 'predict':
        predict_input.main(args)

if __name__ == '__main__':
    main()
