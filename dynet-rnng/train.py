#!/usr/bin/env python
import os
from typing import NamedTuple
import time

import dynet as dy
import numpy as np

from data import Corpus
from trainer import Trainer
from utils import Timer, write_losses, get_folders, write_args


def main(args):
    # Set random seeds.
    np.random.seed(args.seed)
    # TODO: set dynet seed.

    # Create trainer
    trainer = Trainer(
        args=args,
        rnng_type=args.rnng_type,
        name=args.name,
        data_dir=args.data,
        evalb_dir=args.evalb_dir,
        train_path=os.path.join(args.data, 'train/ptb.train.oracle'),
        dev_path=os.path.join(args.data, 'dev/ptb.dev.oracle'),
        test_path=os.path.join(args.data, 'test/ptb.test.oracle'),
        dev_proposal_samples=args.dev_proposal_samples,
        test_proposal_samples=args.test_proposal_samples,
        text_type=args.text_type,
        word_emb_dim=args.word_emb_dim,
        nt_emb_dim=args.nt_emb_dim,
        action_emb_dim=args.action_emb_dim,
        stack_lstm_dim=args.stack_lstm_dim,
        buffer_lstm_dim=args.buffer_lstm_dim,
        terminal_lstm_dim=args.terminal_lstm_dim,
        history_lstm_dim=args.history_lstm_dim,
        lstm_layers=args.lstm_layers,
        composition=args.composition,
        f_hidden_dim=args.f_hidden_dim,
        lr=args.lr,
        lr_decay=args.lr_decay,
        lr_decay_patience=args.lr_decay_patience,
        dropout=args.dropout,
        max_grad_norm=args.clip,
        weight_decay=args.weight_decay,
        use_glove=args.use_glove,
        glove_dir=args.glove_dir,
        fine_tune_embeddings=args.fine_tune_embeddings,
        freeze_embeddings=args.freeze_embeddings,
        print_every=args.print_every,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        elbo_objective=False,
        max_epochs=args.max_epochs,
        max_time=args.max_time,
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
