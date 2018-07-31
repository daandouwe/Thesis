#!/usr/bin/env python
import os
import argparse

import train
import predict

def main():

    parser = argparse.ArgumentParser(description='Discriminative RNNG parser',
                                     fromfile_prefix_chars='@') # Enable loading args from textfile
    # Choose mode
    parser.add_argument('mode', type=str, choices=['train', 'predict'])
    # Data arguments
    parser.add_argument('--data', type=str, default='../tmp',
                        help='location of the data corpus')
    parser.add_argument('--textline', type=str, choices=['unked', 'lower', 'upper'], default='unked',
                        help='textline to use from the oracle file')
    parser.add_argument('--root', type=str, default='.',
                        help='root dir to make output log and checkpoint folders')
    # Model arguments
    parser.add_argument('--char', action='store_true',
                        help='use character-level word embeddings')
    parser.add_argument('--word_emb_dim', type=int, default=100,
                        help='dim of embeddings for word')
    parser.add_argument('--action_emb_dim', type=int, default=100,
                        help='dim of embeddings for actions')
    parser.add_argument('--word_lstm_hidden', type=int, default=100,
                        help='size of lstm hidden states for StackLSTM and BufferLSTM')
    parser.add_argument('--action_lstm_hidden', type=int, default=100,
                        help='size of lstm hidden states for history encoder')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='number of layers in lstm')
    parser.add_argument('--mlp_dim', type=int, default=100,
                        help='size of mlp hidden state')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout rate for embeddings, lstm, and mlp')
    parser.add_argument('--use_glove', action='store_true',
                        help='using pretrained glove embeddings')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed to use')
    # Training arguments
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--clip', type=float, default=5.,
                        help='clipping gradient norm at this value')
    parser.add_argument('--print_every', type=int, default=10,
                        help='when to print training progress')
    parser.add_argument('--disable_cuda', action='store_true',
                        help='disable CUDA')
    args = parser.parse_args()

    if args.mode == 'train':
        train.main(args)
    elif args.mode == 'predict':
        predict.main(args)

if __name__ == '__main__':
    main()
