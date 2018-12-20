import os
import argparse
from math import inf

import train
import predict
import syneval


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
    parser.add_argument('mode', choices=['train', 'predict', 'syneval'],
                        help='what do you want to do?')

    dynet = parser.add_argument_group('Dynet')
    for arg in dynet_args:
        dynet.add_argument(arg)

    data = parser.add_argument_group('Data')
    data.add_argument('--train-path', default='../data/train/ptb.train.trees',
                        help='training trees')
    data.add_argument('--dev-path', default='../data/dev/ptb.dev.trees',
                        help='development trees')
    data.add_argument('--test-path', default='../data/test/ptb.test.trees',
                        help='test trees')
    data.add_argument('--unlabeled-path', default='../data/unlabeled/news.en-00001-of-00100',
                        help='unlabeled data for semi-supervised training')
    data.add_argument('--vocab-path', default=None,
                        help='specify a vocabulary (optional)')
    data.add_argument('--min-word-count', default=None,
                        help='minimal word count')

    model = parser.add_argument_group('Model (shared)')
    model.add_argument('--parser-type', choices=['disc-rnng', 'gen-rnng', 'crf'], required=True,
                        help='type of parser')
    model.add_argument('--model-path-base', required=True,
                        help='path base to use for saving models')
    model.add_argument('--word-emb-dim', type=int, default=100,
                        help='dim of word embeddings')
    model.add_argument('--use-glove', action='store_true',
                        help='using pretrained glove embeddings')
    model.add_argument('--glove-dir', default='~/embeddings/glove',
                        help='to be constructed in main')
    model.add_argument('--fine-tune-embeddings', action='store_true',
                        help='train minimal additive refinement of pretrained embeddings')
    model.add_argument('--freeze-embeddings', action='store_true',
                        help='do not train or fine-tune embeddings')

    crf = parser.add_argument_group('Model (CRF)')
    crf.add_argument('--lstm-dim', type=int, default=250,
                        help='size of lstm hidden')
    crf.add_argument('--lstm-layers', type=int, default=2,
                        help='number of layers in lstm')
    crf.add_argument('--label-hidden-dim', type=int, default=250,
                        help='dimension of label feedforward')

    rnng = parser.add_argument_group('Model (RNNG)')

    rnng.add_argument('--nt-emb-dim', type=int, default=100,
                        help='dim of nonterminal embeddings')
    rnng.add_argument('--action-emb-dim', type=int, default=100,
                        help='dim of nonterminal embeddings')
    rnng.add_argument('--stack-lstm-dim', type=int, default=128,
                        help='size of lstm dim states for stack ecoder')
    rnng.add_argument('--buffer-lstm-dim', type=int, default=128,
                        help='size of lstm hidden states for buffer encoder')
    rnng.add_argument('--terminal-lstm-dim', type=int, default=128,
                        help='size of lstm hidden states for terminal encoder')
    rnng.add_argument('--history-lstm-dim', type=int, default=128,
                        help='size of lstm hidden states for history encoder')
    rnng.add_argument('--composition', default='attention', choices=['basic', 'attention'],
                        help='composition function used by stack-lstm')
    rnng.add_argument('--f-hidden-dim', type=int, default=128,
                        help='dimension of all scoring feedforwards')

    training = parser.add_argument_group('Training')
    training.add_argument('--numpy-seed', type=int, default=42,
                        help='random seed to use')
    training.add_argument('--max-epochs', type=int, default=inf,
                        help='max number of epochs')
    training.add_argument('--max-time', type=int, default=inf,
                        help='max time in seconds')
    training.add_argument('--batch-size', type=int, default=1,
                        help='size of mini batch')
    training.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate for embeddings, lstm, and mlp')
    training.add_argument('--weight-decay', type=float, default=1e-6,
                        help='weight decay (also when using dropout!)')
    training.add_argument('--optimizer', choices=['adam', 'sgd'], default='sgd',
                        help='optimizer used')
    training.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate')
    training.add_argument('--lr-decay', type=float, default=2.,
                        help='anneal lr by lr /= lr-decay')
    training.add_argument('--lr-decay-patience', type=int, default=2,
                        help='wait this many epochs of deteriorating fscore before applying lr-decay')
    training.add_argument('--clip', type=float, default=5.,
                        help='clip gradient if norm is greater than this value')
    training.add_argument('--print-every', type=int, default=10,
                        help='how often to print training progress')
    training.add_argument('--eval-every', type=int, default=-1,
                        help='evaluate model on development set (default: every epoch (-1))')
    training.add_argument('--eval-at-start', action='store_true',
                        help='evaluate model on development set at start of training')
    training.add_argument('--dev-proposal-samples', default='../data/proposal-samples/dev.props',
                        help='proposal samples for development set')
    training.add_argument('--test-proposal-samples', default='../data/proposal-samples/test.props',
                        help='proposal samples for test set')

    semisup = parser.add_argument_group('Semisupervised')
    semisup.add_argument('--joint-model-path', default='checkpoints/joint',
                        help='pretrained joint model (gen-rnng)')
    semisup.add_argument('--post-model-path', default='checkpoints/posterior',
                        help='pretrained posterior model (disc-rnng or crf)')
    semisup.add_argument('--normalize-learning-signal', action='store_true',
                        help='optional baseline')
    semisup.add_argument('--use-argmax-baseline', action='store_true',
                        help='optional baseline')
    semisup.add_argument('--use-mlp-baseline', action='store_true',
                        help='optional baseline')

    # Predict arguments
    prediction = parser.add_argument_group('Prediction')
    prediction.add_argument('--checkpoint', default='',
                        help='load model from this checkpoint')
    prediction.add_argument('--proposal-model', default='',
                        help='load discriminative model (proposal for generative model) from this checkpoint')
    prediction.add_argument('--proposal-samples', default='../data/proposal-samples/dev.props',
                        help='load proposal samples')
    prediction.add_argument('--from-input', action='store_true',
                        help='predict for user input')
    prediction.add_argument('--from-tree-file', action='store_true',
                        help='predict trees for a file of gold trees and evaluate it against those')
    prediction.add_argument('--from-text-file', action='store_true',
                        help='predict trees for a file of tokenized sentences')
    prediction.add_argument('--sample-gen', action='store_true',
                        help='sample from generative model')
    prediction.add_argument('--sample-proposals', action='store_true',
                        help='sample proposals from discriminative model')
    prediction.add_argument('--perplexity', action='store_true',
                        help='evaluate perplexity')
    prediction.add_argument('--inspect-model', action='store_true',
                        help='inspect the attention in the model')
    prediction.add_argument('--num-samples', type=int, default=100,
                        help='number of proposal samples')
    prediction.add_argument('--alpha', type=float, default=1.0,
                        help='temperature to tweak distribution')
    prediction.add_argument('--use-tokenizer', action='store_true',
                        help='tokenize user input')
    prediction.add_argument('--evalb-dir', default='~/EVALB',
                        help='where the evalb excecutable is located')
    prediction.add_argument('--infile', default='.',
                        help='input file to decode')
    prediction.add_argument('--outfile', default='.',
                        help='output file to write to')

    args = parser.parse_args()

    if args.mode == 'train':
        train.main(args)
    elif args.mode == 'predict':
        predict.main(args)
    elif args.mode == 'syneval':
        syneval.main(args)

if __name__ == '__main__':
    main()
