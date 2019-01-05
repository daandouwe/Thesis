#!/usr/bin/env python
import os

import numpy as np

from trainers.supervised import SupervisedTrainer
from trainers.semisupervised import SemiSupervisedTrainer
from trainers.unsupervised import UnsupervisedTrainer
from trainers.wakesleep import WakeSleepTrainer
from trainers.lm import LanguageModelTrainer


def main(args):
    # Set random seeds.
    np.random.seed(args.numpy_seed)

    if args.parser_type in ('disc-rnng', 'gen-rnng', 'crf'):
        trainer = SupervisedTrainer(
            args=args,
            parser_type=args.parser_type,
            model_path_base=args.model_path_base,
            evalb_dir=args.evalb_dir,
            train_path=args.train_path,
            dev_path=args.dev_path,
            test_path=args.test_path,
            vocab_path=args.vocab_path,
            dev_proposal_samples=args.dev_proposal_samples,
            test_proposal_samples=args.test_proposal_samples,
            word_emb_dim=args.word_emb_dim,
            label_emb_dim=args.label_emb_dim,
            action_emb_dim=args.action_emb_dim,
            stack_lstm_dim=args.stack_lstm_dim,
            buffer_lstm_dim=args.buffer_lstm_dim,
            terminal_lstm_dim=args.terminal_lstm_dim,
            history_lstm_dim=args.history_lstm_dim,
            lstm_dim=args.lstm_dim,
            lstm_layers=args.lstm_layers,
            composition=args.composition,
            f_hidden_dim=args.f_hidden_dim,
            label_hidden_dim=args.label_hidden_dim,
            lr=args.lr,
            lr_decay=args.lr_decay,
            lr_decay_patience=args.lr_decay_patience,
            dropout=args.dropout,
            max_grad_norm=args.max_grad_norm,
            weight_decay=args.weight_decay,
            use_glove=args.use_glove,
            glove_dir=args.glove_dir,
            fine_tune_embeddings=args.fine_tune_embeddings,
            freeze_embeddings=args.freeze_embeddings,
            print_every=args.print_every,
            eval_every=args.eval_every,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            max_time=args.max_time,
        )
    elif args.parser_type in ('semisup-rnng', 'semisup-crf'):
        trainer = SemiSupervisedTrainer(
            args=args,
            evalb_dir=args.evalb_dir,
            train_path=args.train_path,
            dev_path=args.dev_path,
            test_path=args.test_path,
            unlabeled_path=args.unlabeled_path,
            joint_model_path=args.joint_model_path,
            post_model_path=args.post_model_path,
            use_argmax_baseline=args.use_argmax_baseline,
            use_mlp_baseline=args.use_mlp_baseline,
            lmbda=1.0,
            clip_learning_signal=None,
            num_samples=args.num_samples,
            alpha=args.alpha,
            lr=args.lr,
            lr_decay=args.lr_decay,
            lr_decay_patience=args.lr_decay_patience,
            max_grad_norm=args.max_grad_norm,
            weight_decay=args.weight_decay,
            use_glove=args.use_glove,
            glove_dir=args.glove_dir,
            print_every=args.print_every,
            eval_every=args.eval_every,
            eval_at_start=args.eval_at_start,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            max_time=args.max_time,
        )
    elif args.parser_type in ('unsup-rnng', 'unsup-crf'):
        trainer = UnsupervisedTrainer(
            args=args,
            evalb_dir=args.evalb_dir,
            train_path=args.unlabeled_path,
            dev_path=args.dev_path,
            test_path=args.test_path,
            min_word_count=5,
            max_sent_len=40,
            num_labels=30,
            use_argmax_baseline=args.use_argmax_baseline,
            use_mlp_baseline=args.use_mlp_baseline,
            clip_learning_signal=None,
            num_samples=args.num_samples,
            alpha=args.alpha,
            lr=args.lr,
            lr_decay=args.lr_decay,
            lr_decay_patience=args.lr_decay_patience,
            max_grad_norm=args.max_grad_norm,
            weight_decay=args.weight_decay,
            use_glove=args.use_glove,
            glove_dir=args.glove_dir,
            print_every=args.print_every,
            eval_every=args.eval_every,
            eval_at_start=args.eval_at_start,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            max_time=args.max_time,
        )
    elif args.parser_type == 'rnn-lm':
        trainer = LanguageModelTrainer(
            model_path_base=args.model_path_base,
            multitask=args.multitask,
            predict_all_spans=args.all_spans,
            args=args,
            train_path=args.train_path,
            dev_path=args.dev_path,
            test_path=args.test_path,
            vocab_path=args.vocab_path,
            emb_dim=args.word_emb_dim,
            lstm_dim=args.lstm_dim,
            lstm_layers=args.lstm_layers,
            label_hidden_dim=args.label_hidden_dim,
            max_epochs=args.max_epochs,
            max_time=args.max_time,
            lr=args.lr,
            batch_size=args.batch_size,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            lr_decay=args.lr_decay,
            lr_decay_patience=args.lr_decay_patience,
            max_grad_norm=args.max_grad_norm,
            use_glove=args.use_glove,
            glove_dir=args.glove_dir,
            fine_tune_embeddings=args.fine_tune_embeddings,
            freeze_embeddings=args.freeze_embeddings,
            print_every=args.print_every,
            eval_every=args.eval_every,
        )

    # Train the model
    trainer.train()

    # Move the folder to it's final place
    trainer.finalize_model_folder()
