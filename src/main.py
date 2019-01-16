import os
import argparse
from math import inf

import build
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
                                     fromfile_prefix_chars='@') # enables loading args from textfile
    # Choose mode
    parser.add_argument('mode', choices=['build', 'train', 'predict', 'syneval'],
                        help='what do you want to do?')

    dynet = parser.add_argument_group('Dynet')
    for arg in dynet_args:
        dynet.add_argument(arg)

    data = parser.add_argument_group('Data')
    data.add_argument('--train-path', default='data/ptb/train.trees',
                        help='training trees')
    data.add_argument('--dev-path', default='data/ptb/dev.trees',
                        help='development trees')
    data.add_argument('--test-path', default='data/ptb/test.trees',
                        help='test trees')
    data.add_argument('--unlabeled-path', default='data/unlabeled/news.en-00001-of-00100.processed',
                        help='unlabeled data for semi-supervised training')
    data.add_argument('--vocab-path', default=None,
                        help='specify a vocabulary (optional)')

    vocab = parser.add_argument_group('Vocabulary')
    vocab.add_argument('--min-word-count', type=int, default=None,
                        help='minimal word count for vocab')
    vocab.add_argument('--max-vocab-size', type=int, default=-1,
                        help='maxinum number of words in vocab')
    vocab.add_argument('--lowercase', action='store_true',
                        help='lowercase vocab')

    model = parser.add_argument_group('Model (shared)')
    model.add_argument('--model-type', choices=['disc-rnng', 'gen-rnng', 'crf', 'semisup-crf', 'semisup-disc', 'fully-unsup-disc', 'rnn-lm'],
                        help='type of model', default='disc-rnng')
    model.add_argument('--model-path-base', default='disc-rnng',
                        help='path base to use for saving models')
    model.add_argument('--word-emb-dim', type=int, default=100,
                        help='dim of word embeddings')
    model.add_argument('--use-glove', action='store_true',
                        help='using pretrained glove embeddings')
    model.add_argument('--glove-dir', default='~/embeddings/glove',
                        help='location of glove embeddings')
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

    rnng.add_argument('--label-emb-dim', type=int, default=100,
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

    lm = parser.add_argument_group('Model (LM)')
    lm.add_argument('--multitask', action='store_true',
                    help='predict labeled spans as side objective')
    lm.add_argument('--all-spans', action='store_true',
                    help='also predict null spans')

    training = parser.add_argument_group('Train')
    training.add_argument('--numpy-seed', type=int, default=42,
                        help='random seed to use')
    training.add_argument('--max-epochs', type=int, default=inf,
                        help='max number of epochs')
    training.add_argument('--max-time', type=int, default=inf,
                        help='max time in seconds')
    training.add_argument('--max-lines', type=int, default=-1,
                        help='max lines from the dataset')
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
    training.add_argument('--max-grad-norm', type=float, default=5.,
                        help='clip gradient if norm is greater than this value')
    training.add_argument('--print-every', type=int, default=10,
                        help='how often to print training progress')
    training.add_argument('--eval-every', type=int, default=-1,
                        help='evaluate model on development set each number of updates (default: every epoch (-1))')
    training.add_argument('--eval-every-epochs', type=int, default=1,
                        help='evaluate model on development set each number of epochs (for gen-rnng)')
    training.add_argument('--eval-at-start', action='store_true',
                        help='evaluate model on development set at start of training (for semisupervised)')
    training.add_argument('--dev-proposal-samples', default='data/proposals/rnng-dev.props',
                        help='proposal samples for development set')
    training.add_argument('--test-proposal-samples', default='data/proposals/rnng-test.props',
                        help='proposal samples for test set')
    training.add_argument('--num-dev-samples', type=int, default=20,
                        help='number of samples to estimate the development fscore and perplexity')
    training.add_argument('--num-test-samples', type=int, default=100,
                        help='number of samples to estimate the test fscore and perplexity')

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
    pred = parser.add_argument_group('Predict')
    pred.add_argument('--checkpoint', default='',
                        help='load model from this checkpoint')
    pred.add_argument('--infile', default='',
                        help='input file to decode')
    pred.add_argument('--outfile', default='',
                        help='output file to write to')
    pred.add_argument('--indir', default='',
                        help='input directory (primarily syneval)')
    pred.add_argument('--outdir', default='',
                        help='output directory to write to (primarily syneval)')
    pred.add_argument('--from-input', action='store_true',
                        help='predict for user input')
    pred.add_argument('--from-tree-file', action='store_true',
                        help='predict trees for a file of gold trees and evaluate it against those')
    pred.add_argument('--from-text-file', action='store_true',
                        help='predict trees for a file of tokenized sentences')
    pred.add_argument('--use-tokenizer', action='store_true',
                        help='tokenize user input')
    pred.add_argument('--sample-gen', action='store_true',
                        help='sample from generative model')
    pred.add_argument('--sample-proposals', action='store_true',
                        help='sample proposals from discriminative model')
    pred.add_argument('--num-samples', type=int, default=100,
                        help='number of proposal samples')
    pred.add_argument('--alpha', type=float, default=1.0,
                        help='temperature to tweak distribution')
    pred.add_argument('--perplexity', action='store_true',
                        help='evaluate perplexity')
    pred.add_argument('--proposal-model', default='',
                        help='load discriminative model (proposal for generative model) from this checkpoint')
    pred.add_argument('--proposal-samples', default='data/proposals/rnng-dev.props',
                        help='load proposal samples')
    pred.add_argument('--inspect-model', action='store_true',
                        help='inspect the attention in the model')
    pred.add_argument('--evalb-dir', default='EVALB',
                        help='where the evalb excecutable is located')

    syn = parser.add_argument_group('Syneval')
    syn.add_argument('--syneval-short', action='store_true',
                        help='evaluate on a subset of syneval (only the small datasets)')
    syn.add_argument('--syneval-max-lines', type=int, default=-1,
                        help='subsample the syneval dataset if it exceeds this (especially for gen-rnng)')
    syn.add_argument('--capitalize', action='store_true',
                        help='capitalize the sentence (especially for syneval)')
    syn.add_argument('--add-period', action='store_true',
                        help='add a period at the end of the sentence (especially for syneval)')

    args = parser.parse_args()

    if args.mode == 'build':
        build.main(args)
    if args.mode == 'train':
        train.main(args)
    elif args.mode == 'predict':
        predict.main(args)
    elif args.mode == 'syneval':
        syneval.main(args)


if __name__ == '__main__':
    main()
