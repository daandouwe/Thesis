import os

import numpy as np

from trainer import SemiSupervisedTrainer


def main(args):
    # Set random seeds.
    np.random.seed(args.seed)

    # Create trainer
    trainer = SemiSupervisedTrainer(
        args=args,
        name=args.name,
        data_dir=args.data,
        evalb_dir=args.evalb_dir,
        train_path=os.path.join(args.data, 'train/ptb.train.oracle'),
        dev_path=os.path.join(args.data, 'dev/ptb.dev.oracle'),
        test_path=os.path.join(args.data, 'test/ptb.test.oracle'),
        lm_path='/Users/daan/data/one-billion-words/heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100',
        text_type=args.text_type,
        joint_model_path='checkpoints/joint',
        post_model_path='checkpoints/posterior',
        use_mean_baseline=False,
        use_argmax_baseline=True,
        use_lm_baseline=False,
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
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        max_time=args.max_time,
    )

    # Train the model
    try:
        trainer.train()
    except KeyboardInterrupt:
        print('-'*99)
        print('Exiting from training early.')
