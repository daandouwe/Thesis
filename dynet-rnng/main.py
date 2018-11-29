#!/usr/bin/env python
import os
import argparse
from math import inf

import train
import predict
import semisup
import unsup
import wakesleep


def main():

    dynet_args = [
        '--dynet-mem',
        '--dynet-weight-decay',
        '--dynet-autobatch',
        '--dynet-gpus',
        '--dynet-gpu',
        '--dynet-devices',
        '--dynet-seed',
    ]

    parser = argparse.ArgumentParser(description='RNNG parser',
                                     fromfile_prefix_chars='@') # enable loading args from textfile
    # Choose mode
    parser.add_argument('mode', choices=['train', 'predict', 'inspect', 'latent', 'semisup', 'wakesleep', 'unsup'],
                        help='what do you want to do?')
    parser.add_argument('rnng_type', choices=['disc', 'gen'],
                        help='use discriminative or generative model')

    for arg in dynet_args:
        parser.add_argument(arg)

    # Debugging
    parser.add_argument('-d', '--debug', action='store_true')

    # Data arguments
    parser.add_argument('--train-path', default='../data/train/ptb.train.trees',
                        help='training trees')
    parser.add_argument('--dev-path', default='../data/dev/ptb.dev.trees',
                        help='development trees')
    parser.add_argument('--test-path', default='../data/test/ptb.test.trees',
                        help='test trees')
    parser.add_argument('--unlabeled-path', default='../data/unlabeled/news.en-00001-of-00100',
                        help='unlabeled data for semi-supervised training')
    parser.add_argument('--vocab-path', default=None,
                        help='specify a vocabulary (optional)')
    parser.add_argument('--root', default='.',
                        help='root dir to make output log and checkpoint folders')
    parser.add_argument('--disable-subdir', action='store_true',
                        help='do not make subdirectory inside `logdir`, `checkdir` and `outdir`')
    parser.add_argument('--logdir', default='log',
                        help='directory for logs')
    parser.add_argument('--outdir', default='out',
                        help='directory for predictions')
    parser.add_argument('--checkdir', default='checkpoints',
                        help='directory to save models')
    parser.add_argument('--disable-folders', action='store_true',
                        help='do not make output folders (debug)')
    parser.add_argument('--max-lines', default=-1, type=int,
                        help='max number of training lines')

    # Model arguments
    parser.add_argument('--word-emb-dim', type=int, default=100,
                        help='dim of word embeddings')
    parser.add_argument('--nt-emb-dim', type=int, default=100,
                        help='dim of nonterminal embeddings')
    parser.add_argument('--action-emb-dim', type=int, default=20,
                        help='dim of nonterminal embeddings')
    parser.add_argument('--stack-lstm-dim', type=int, default=128,
                        help='size of lstm dim states for stack ecoder')
    parser.add_argument('--buffer-lstm-dim', type=int, default=128,
                        help='size of lstm hidden states for buffer encoder')
    parser.add_argument('--terminal-lstm-dim', type=int, default=128,
                        help='size of lstm hidden states for terminal encoder')
    parser.add_argument('--history-lstm-dim', type=int, default=32,
                        help='size of lstm hidden states for history encoder')
    parser.add_argument('--lstm-layers', type=int, default=2,
                        help='number of layers in lstm')
    parser.add_argument('--composition', default='attention', choices=['basic', 'attention', 'latent-factors'],
                        help='composition function used by StackLSTM')
    parser.add_argument('--f-hidden-dim', type=int, default=128,
                        help='hidden dimension of scoring feedforward')
    parser.add_argument('--use-glove', action='store_true',
                        help='using pretrained glove embeddings')
    parser.add_argument('--glove-dir', default='~/embeddings/glove',
                        help='to be constructed in main')
    parser.add_argument('--fine-tune-embeddings', action='store_true',
                        help='train minimal additive refinement of pretrained embeddings')
    parser.add_argument('--freeze-embeddings', action='store_true',
                        help='do not train or fine-tune embeddings')

    # Training arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed to use')
    parser.add_argument('--max-epochs', type=int, default=inf,
                        help='max number of epochs')
    parser.add_argument('--max-time', type=int, default=inf,
                        help='max time in seconds')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='size of mini batch')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate for embeddings, lstm, and mlp')
    parser.add_argument('--weight-decay', type=float, default=1e-6,
                        help='weight decay (also when using dropout!)')
    parser.add_argument('--optimizer', choices=['adam', 'sgd', 'rmsprop'], default='sgd',
                        help='optimizer used')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--lr-decay', type=float, default=2.,
                        help='anneal lr by lr /= lr-decay')
    parser.add_argument('--lr-decay-patience', type=int, default=2,
                        help='waiting epochs of deteriorating fscore before applying lr-decay')
    parser.add_argument('--momentum', type=float, default=0,
                        help='momentum for sgd')
    parser.add_argument('--disable-kl-anneal', action='store_false',
                        help='do not anneal the kl in the elbo objective')
    parser.add_argument('--disable-glorot', action='store_true',
                        help='do not override custom lstm initialization with glorot')
    parser.add_argument('--clip', type=float, default=5.,
                        help='clipping gradient norm at this value')
    parser.add_argument('--print-every', type=int, default=10,
                        help='when to print training progress')
    parser.add_argument('--eval-every', type=int, default=-1,
                        help='evaluate model on development set (default: every epoch (-1))')
    parser.add_argument('--eval-at-start', action='store_true',
                        help='evaluate model on development set at start of training')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable cuda')
    parser.add_argument('--num-procs', type=int, default=1,
                        help='number of processes to spawn for parallel training')
    parser.add_argument('--dev-proposal-samples', default='../data/proposal-samples/dev.props',
                        help='proposal samples for development set')
    parser.add_argument('--test-proposal-samples', default='../data/proposal-samples/test.props',
                        help='proposal samples for test set')

    # Semi-supervised arguments
    parser.add_argument('--joint-model-path', default='checkpoints/joint',
                        help='pretrained joint model (GenRNNG)')
    parser.add_argument('--post-model-path', default='checkpoints/posterior',
                        help='pretrained posterior model (DiscRNNG)')
    parser.add_argument('--use-argmax-baseline', action='store_true')

    parser.add_argument('--use-mlp-baseline', action='store_true')

    # Predict arguments
    parser.add_argument('--checkpoint', default='',
                        help='load model from this checkpoint')
    parser.add_argument('--proposal-model', default='',
                        help='load discriminative model (proposal for generative model) from this checkpoint')
    parser.add_argument('--proposal-samples', default='../data/proposal-samples/dev.props',
                        help='load proposal samples')
    parser.add_argument('--from-input', action='store_true',
                        help='predict for user input')
    parser.add_argument('--from-tree-file', action='store_true',
                        help='predict trees for a file of gold trees and evaluate it against those')
    parser.add_argument('--from-text-file', action='store_true',
                        help='predict trees for a file of tokenized sentences')
    parser.add_argument('--sample-gen', action='store_true',
                        help='sample from generative model')
    parser.add_argument('--sample-proposals', action='store_true',
                        help='sample proposals from discriminative model')
    parser.add_argument('--perplexity', action='store_true',
                        help='evaluate perplexity')
    parser.add_argument('--syneval', action='store_true',
                        help='evaluate on syneval test')
    parser.add_argument('--inspect-model', action='store_true',
                        help='inspect the attention in the model')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='number of proposal samples')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='temperature to tweak distribution')
    parser.add_argument('--use-tokenizer', action='store_true',
                        help='tokenize user input')
    parser.add_argument('--evalb-dir', default='~/EVALB',
                        help='where the evalb excecutable is located')
    parser.add_argument('--infile', default='.',
                        help='input file to decode')
    parser.add_argument('--outfile', default='.',
                        help='output file to write to')

    # Latent model arguments:
    parser.add_argument('--observation-model', choices=['bow', 'rnn', 'heads-rnn', 'crf'], default='bow',
                        help='type of observation model')
    parser.add_argument('--latent-dim', type=int, default=100,
                        help='dimension of latent space')
    parser.add_argument('--word-dropout', type=float, default=0.,
                        help='word dropout for rnn observation model')
    parser.add_argument('--tree-dropout', action='store_true',
                        help='dynamic dropout for stack representation')
    parser.add_argument('--use-gating', action='store_true',
                        help='use gated score for word logits')

    args = parser.parse_args()

    if args.mode == 'train':
        train.main(args)
    elif args.mode == 'predict':
        predict.main(args)
    elif args.mode == 'semisup':
        semisup.main(args)
    elif args.mode == 'unsup':
        unsup.main(args)
    elif args.mode == 'wakesleep':
        wakesleep.main(args)


if __name__ == '__main__':
    main()
