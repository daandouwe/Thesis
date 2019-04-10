#!/usr/bin/env python
import os

import numpy as np

from trainers.supervised import SupervisedTrainer
from trainers.semisupervised import SemiSupervisedTrainer
from trainers.unsupervised import FullyUnsupervisedTrainer
from trainers.wakesleep import WakeSleepTrainer
from trainers.lm import LanguageModelTrainer


def main(args):
    # Set random seeds.
    np.random.seed(args.numpy_seed)

    if args.model_type in ('disc-rnng', 'gen-rnng', 'crf'):
        trainer = SupervisedTrainer(
            args=args,
            model_type=args.model_type,
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
            batch_size=args.batch_size,
            optimizer_type=args.optimizer,
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
            eval_every_epochs=args.eval_every_epochs,
            max_epochs=args.max_epochs,
            max_time=args.max_time,
            num_dev_samples=args.num_dev_samples,
            num_test_samples=args.num_test_samples,
        )
    elif args.model_type in ('semisup-disc', 'semisup-crf'):
        trainer = SemiSupervisedTrainer(
            args=args,
            model_type=args.model_type,
            model_path_base=args.model_path_base,
            evalb_dir=args.evalb_dir,
            train_path=args.train_path,
            dev_path=args.dev_path,
            test_path=args.test_path,
            unlabeled_path=args.unlabeled_path,
            joint_model_path=args.joint_model_path,
            post_model_path=args.post_model_path,
            lmbda=args.lmbda,
            use_argmax_baseline=args.use_argmax_baseline,
            use_mlp_baseline=args.use_mlp_baseline,
            exact_entropy=args.exact_entropy,
            clip_learning_signal=None,
            num_samples=args.num_samples,
            alpha=args.alpha,
            batch_size=args.batch_size,
            optimizer_type=args.optimizer,
            lr=args.lr,
            lr_decay=args.lr_decay,
            lr_decay_patience=args.lr_decay_patience,
            max_grad_norm=args.max_grad_norm,
            weight_decay=args.weight_decay,
            print_every=args.print_every,
            eval_every=args.eval_every,
            eval_at_start=args.eval_at_start,
            max_epochs=args.max_epochs,
            max_time=args.max_time,
            num_dev_samples=args.num_dev_samples,
            num_test_samples=args.num_test_samples,
        )
    elif args.model_type in ('fully-unsup-disc', 'fully-unsup-crf'):
        trainer = FullyUnsupervisedTrainer(
            args=args,
            model_type=args.model_type,
            model_path_base=args.model_path_base,
            evalb_dir=args.evalb_dir,
            train_path=args.train_path,
            dev_path=args.dev_path,
            test_path=args.test_path,
            vocab_path=args.vocab_path,
            num_labels=10,
            use_argmax_baseline=args.use_argmax_baseline,
            use_mlp_baseline=args.use_mlp_baseline,
            clip_learning_signal=None,
            num_samples=args.num_samples,
            alpha=args.alpha,
            batch_size=args.batch_size,
            optimizer_type=args.optimizer,
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
            max_epochs=args.max_epochs,
            max_time=args.max_time,
            num_dev_samples=args.num_dev_samples,
            num_test_samples=args.num_test_samples,
        )
    elif args.model_type == 'rnn-lm':
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
    else:
        raise ValueError(f'Invalid model {args.model_type}.')

    if args.resume:
        # load all settings from checkpoint
        trainer.load_state_to_resume(args.resume)

    # Train the model
    trainer.train()

    # Move the folder to it's final place
    trainer.finalize_model_folder()
