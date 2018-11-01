#!/usr/bin/env python
import os
from typing import NamedTuple
import time

import dynet as dy
import numpy as np

from data import Corpus
from model import DiscRNNG, GenRNNG
from trainer import Trainer
from utils import Timer, write_losses, get_folders, write_args

DEBUG_NUM_LINES = 10


def main(args):
    # Set random seeds.
    np.random.seed(args.seed)

    # Make output folder structure.
    subdir, logdir, checkdir, outdir = get_folders(args)
    print(f'Output subdirectory: `{subdir}`.')
    print(f'Saving logs to `{logdir}`.')
    print(f'Saving predictions to `{outdir}`.')
    print(f'Saving models to `{checkdir}`.')
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(checkdir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    # Save arguments.
    write_args(args, logdir)

    if args.debug:
        args.max_lines = DEBUG_NUM_LINES

    # Get data
    corpus = Corpus(
        train_path=os.path.join(args.data, 'train/ptb.train.oracle'),
        dev_path=os.path.join(args.data, 'dev/ptb.dev.oracle'),
        test_path=os.path.join(args.data, 'test/ptb.test.oracle'),
        text=args.text,
        model=args.model
    )

    # Create model
    model = dy.Model()
    if args.model == 'disc':
        rnng = DiscRNNG(
            model=model,
            dictionary=corpus.dictionary,
            num_words=len(corpus.dictionary.w2i),
            num_nt=len(corpus.dictionary.n2i),
            word_emb_dim=args.word_emb_dim,
            nt_emb_dim=args.nt_emb_dim,
            action_emb_dim=args.action_emb_dim,
            stack_hidden_size=args.stack_lstm_hidden,
            buffer_hidden_size=args.buffer_lstm_hidden,
            history_hidden_size=args.history_lstm_hidden,
            stack_num_layers=args.lstm_num_layers,
            buffer_num_layers=args.lstm_num_layers,
            history_num_layers=args.lstm_num_layers,
            composition=args.composition,
            mlp_hidden=args.mlp_hidden,
            dropout=args.dropout
        )
    elif args.model == 'gen':
        rnng = GenRNNG(
            model=model,
            dictionary=corpus.dictionary,
            num_words=len(corpus.dictionary.w2i),
            num_nt=len(corpus.dictionary.n2i),
            word_emb_dim=args.word_emb_dim,
            nt_emb_dim=args.nt_emb_dim,
            action_emb_dim=args.action_emb_dim,
            stack_hidden_size=args.stack_lstm_hidden,
            terminal_hidden_size=args.terminal_lstm_hidden,
            history_hidden_size=args.history_lstm_hidden,
            stack_num_layers=args.lstm_num_layers,
            terminal_num_layers=args.lstm_num_layers,
            history_num_layers=args.lstm_num_layers,
            composition=args.composition,
            mlp_hidden=args.mlp_hidden,
            dropout=args.dropout
        )

    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = dy.SimpleSGDTrainer(model, learning_rate=args.lr)
    elif args.optimizer == 'adam':
        optimizer = dy.AdamTrainer(model, alpha=args.lr)
    optimizer.set_clip_threshold(args.clip)
    model.set_weight_decay(args.weight_decay)

    # Create trainer
    trainer = Trainer(
        rnng_type=args.model,
        model=rnng,
        dictionary=corpus.dictionary,
        optimizer=optimizer,
        train_dataset=corpus.train.data,
        dev_dataset=corpus.dev.data,
        test_dataset=corpus.test.data,
        dev_proposal_samples=args.dev_proposal_samples,
        test_proposal_samples=args.test_proposal_samples,
        lr=args.lr,
        fine_tune_embeddings=args.fine_tune_embeddings,
        print_every=args.print_every,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        elbo_objective=False,
        max_epochs=args.max_epochs,
        max_time=args.max_time,
        name=args.name,
        checkpoint_dir=checkdir,
        output_dir=outdir,
        log_dir=logdir,
        data_dir=args.data,
        evalb_dir=args.evalb_dir,
        args=args,
    )

    # Train the model
    try:
        trainer.train()
    except KeyboardInterrupt:
        print('-'*99)
        print('Exiting from training early.')

    trainer.check_dev()

    # Save the losses for plotting and diagnostics
    trainer.write_losses()

    test_fscore = trainer.check_test()
    print('='*99)
    print('| End of training | Best dev F1 {:3.2f} (epoch {:2d}) | Test F1 {:3.2f}'.format(
        trainer.best_dev_fscore, trainer.best_dev_epoch, test_fscore))
    print('='*99)
    # Save model again but with test fscore
    trainer.test_fscore = test_fscore
    trainer.save_checkpoint()
