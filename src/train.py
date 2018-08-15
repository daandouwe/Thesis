import gc
import itertools

import numpy as np
import torch
from tensorboardX import SummaryWriter

from data_test import Corpus
from model_test import make_model
from predict import predict
from eval import evalb
from util import Timer, write_losses, make_folders


# gc.set_debug(gc.DEBUG_LEAK)


def schedule_lr(args, optimizer, update):
    update = update + 1
    warmup_coeff = args.lr / args.learning_rate_warmup_steps
    if update <= args.learning_rate_warmup_steps:
        for param_group in optimizer.param_groups:
            param_group['lr'] = update * warmup_coeff


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def batchify(batches, batch_size):
    def ceil_div(a, b):
        return ((a-1) // b) + 1
    return [batches[i*batch_size:(i+1)*batch_size]
            for i in range(ceil_div(len(batches), batch_size))]


def main(args):
    # Set random seeds.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Set cuda.
    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Device: {args.device}.')

    if not args.disable_folders:
        make_folders(args)
    else:
        print('Did not make output folders!')

    print(f'Created tensorboard summary writer at {args.logdir}.')
    writer = SummaryWriter(args.logdir)

    print(f'Loading data from {args.data}...')
    corpus = Corpus(data_path=args.data, textline=args.textline, name_template=args.name_template, char=args.use_char)
    train_batches = corpus.train.batches(length_ordered=False, shuffle=True)
    dev_batches = corpus.dev.batches(length_ordered=False, shuffle=False)
    test_batches = corpus.test.batches(length_ordered=False, shuffle=False)
    print(corpus)

    if args.debug:
        print('Debug mode.')
        train_batches = train_batches[:30]
        dev_batches = dev_batches
        test_batches = test_batches

    model = make_model(args, corpus.dictionary)
    model.to(args.device)

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    # Learning rate is set during training by set_lr().
    optimizer = torch.optim.Adam(trainable_parameters, lr=1., betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max',
        factor=args.step_decay_factor,
        patience=args.step_decay_patience,
        verbose=True,
    )

    print('Training...')
    losses = list()
    num_updates = 0
    best_dev_fscore = -np.inf
    best_dev_epoch = None
    test_fscore = None

    def save_checkpoint():
        with open(args.checkfile, 'wb') as f:
            # Could move to using 'model': model.state_dict()
            # Then:
            # model_state_dict = state['model']
            # args = state['args']
            # dictionary = state['dictionary']
            # model = load_model(args, dictionary)
            # model.load_state_dict(model_state_dict)
            state = {
                'args': args,
                'model': model,
                'dictionary': corpus.dictionary,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'epoch': epoch,
                'num-updates': num_updates,
                'best-dev-fscore': best_dev_fscore,
                'best-dev-epoch': best_dev_epoch,
                'test-fscore': test_fscore
            }
            torch.save(state, f)

    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_epoch

        model.eval()
        predict(model, dev_batches, args.outdir, name='dev')
        dev_fscore = evalb(args.outdir, args.data, args.name_template, name='dev')
        writer.add_scalar('Dev/Fscore', dev_fscore, num_updates)
        if dev_fscore > best_dev_fscore:
            print(f'Saving new best model to {args.checkfile}...')
            save_checkpoint()
            best_dev_epoch = epoch
            best_dev_fscore = dev_fscore
        return dev_fscore

    def train_epoch():
        """One epoch of training."""
        nonlocal num_updates
        nonlocal losses

        model.train()
        train_timer = Timer()
        num_sentences = len(train_batches)
        num_batches = num_sentences // args.batch_size
        processed = 0
        for step, minibatch in enumerate(batchify(train_batches, args.batch_size), 1):
            # Set learning rate.
            num_updates += 1
            processed += args.batch_size
            schedule_lr(args, optimizer, num_updates)

            # Compute loss over minibatch.
            loss = torch.zeros(1, device=args.device)
            for batch in minibatch:
                sentence, actions = batch
                loss += model(sentence, actions)
            loss /= args.batch_size

            # TODO: Look into Adam optimizer if memory use is growing.

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, args.clip)
            optimizer.step()

            loss = loss.item()
            losses.append(loss)
            if step % args.print_every == 0:
                # Log to tensorboard.
                writer.add_scalar('Train/Loss', loss, num_updates)
                writer.add_scalar('Train/Learning-rate', get_lr(optimizer), num_updates)
                avg_loss = np.mean(losses[-args.print_every:])
                lr = get_lr(optimizer)
                sents_per_sec = processed / train_timer.elapsed()
                eta = (num_sentences - processed) / sents_per_sec
                print(
                    f'| step {step:6d}/{num_batches:5d} '
                    f'| loss {avg_loss:7.3f} '
                    f'| lr {lr:.1e} '
                    f'| {sents_per_sec:4.1f} sents/sec '
                    f'| eta {train_timer.format(eta)}'
                )



    epoch_timer = Timer()
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        # No upper limit of epochs
        for epoch in itertools.count(start=1):
            if args.epochs is not None and epoch > args.epochs:
                break

            # Shuffle batches each epoch.
            np.random.shuffle(train_batches)

            # Train one epoch.
            train_epoch()

            print('Evaluating fscore on development set...')
            dev_fscore = check_dev()

            # Scheduler for learning rate.
            if args.step_decay:
                if (num_updates // args.batch_size + 1) > args.learning_rate_warmup_steps:
                    scheduler.step(best_dev_fscore)

            print('-'*89)
            print(
                f'| End of epoch {epoch:3d}/{args.epochs} '
                f'| total-elapsed {epoch_timer.format_elapsed()}'
                f'| dev-fscore {dev_fscore:4.2f} '
                f'| best dev-epoch {best_dev_epoch} '
                f'| best dev-fscore {best_dev_fscore:4.2f} '
            )
            print('-'*89)
    except KeyboardInterrupt:
        print('-'*89)
        print('Exiting from training early.')
        # Save the losses for plotting and diagnostics.
        write_losses(args, losses)
        # TODO(not sure) writer.export_scalars_to_json('scalars.json')
        print('Evaluating fscore on development set...')
        check_dev()
    # Load best saved model.
    print(f'Loading best saved model (epoch {best_dev_epoch}) from {args.checkfile}...')
    with open(args.checkfile, 'rb') as f:
        state = torch.load(f)
        model = state['model']

    print('Evaluating loaded model on test set...')
    predict(model, test_batches, args.outdir, name='test')
    test_fscore = evalb(args.outdir, args.data, args.name_template, name='test')
    save_checkpoint()

    print('-'*89)
    print(
         f'| End of training '
         f'| best dev-epoch {best_dev_epoch:2d} '
         f'| best dev-fscore {best_dev_fscore:4.2f} '
         f'| test-fscore {test_fscore}'
    )
    print('-'*89)
