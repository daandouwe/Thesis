import os

import numpy as np

from trainer import SemiSupervisedTrainer


def main(args):
    # Set random seeds.
    np.random.seed(args.seed)

    # Create trainer
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
        max_grad_norm=args.clip,
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

    # Train the model
    trainer.train()
