import os
import itertools
import multiprocessing as mp

import numpy as np
import torch
from tensorboardX import SummaryWriter

from data import Corpus
from model import make_model
from trainer import Trainer
from eval import evalb
from utils import Timer, write_losses, get_folders, write_args


DEBUG_NUM_LINES = 10


def main(args):
    # Set random seeds.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set cuda.
    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Device: {args.device}.')

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

    print(f'Loading data from `{args.data}`...')
    corpus = Corpus(
        data_path=args.data,
        model=args.model,
        textline=args.textline,
        name=args.name,
        use_chars=args.use_chars,
        max_lines=args.max_lines
    )
    train_dataset = corpus.train.batches(shuffle=True)
    dev_dataset = corpus.dev.batches()
    test_dataset = corpus.test.batches()
    print(corpus)

    # Sometimes we don't want to use all data.
    if args.debug:
        print('Debug mode.')
        dev_dataset = dev_dataset[:DEBUG_NUM_LINES]
        test_dataset = test_dataset[:DEBUG_NUM_LINES]
    elif args.max_lines != -1:
        dev_dataset = dev_dataset[:100]
        test_dataset = test_dataset[:100]

    # Create model.
    model = make_model(args, corpus.dictionary)
    model.to(args.device)

    # Do we have a KL term the loss?
    elbo_objective = (args.composition in ('latent-factors', 'latent-attention'))

    # Choose optimizer.
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if args.optimizer == 'adam':
        name = 'Adam'
        optimizer = torch.optim.Adam(
            trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        name = 'SGD'
        optimizer = torch.optim.SGD(
            trainable_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        name = 'RMSprop'
        optimizer = torch.optim.RMSprop(
            trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    print(f'Using {name} optimizer with initial learning rate {args.lr} and momentum {args.momentum}.')

    # Training minibatches accross multiple processors?
    num_procs = mp.cpu_count() if args.num_procs == -1 else args.num_procs

    trainer = Trainer(
        rnng_type = args.model,
        model=model,
        dictionary=corpus.dictionary,
        optimizer=optimizer,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        dev_proposal_samples=args.dev_proposal_samples,
        test_proposal_samples=args.test_proposal_samples,
        num_procs=num_procs,
        lr=args.lr,
        fine_tune_embeddings=args.fine_tune_embeddings,
        print_every=args.print_every,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        elbo_objective=elbo_objective,
        max_epochs=args.max_epochs,
        max_time=args.max_time,
        name=args.name,
        checkpoint_dir=checkdir,
        output_dir=outdir,
        log_dir=logdir,
        data_dir=args.data,
        evalb_dir=args.evalb_dir,
        device=args.device,
        max_grad_norm=args.clip,
        args=args,
    )

    # Train the model.
    print(f'Training with {num_procs} processes...')
    try:
        trainer.train()
    except KeyboardInterrupt:
        print('-'*99)
        print('Exiting from training early.')

    trainer.check_dev()

    # Save the losses for plotting and diagnostics.
    trainer.write_losses()

    test_fscore = trainer.check_test()
    print('='*99)
    print('| End of training | Best dev F1 {:3.2f} (epoch {:2d}) | Test F1 {:3.2f}'.format(
        trainer.best_dev_fscore, trainer.best_dev_epoch, test_fscore))
    print('='*99)
    # Save model again but with test fscore.
    trainer.test_fscore = test_fscore
    trainer.save_checkpoint()
