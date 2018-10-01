import os
import itertools
from math import inf

import numpy as np
import torch
from tensorboardX import SummaryWriter

from data import Corpus
from model import make_model
from decode import GreedyDecoder
from eval import evalb
from utils import Timer, get_folders, write_args


class Trainer:
    """Trainer for RNNG."""
    def __init__(
        self,
        model=None,
        optimizer=None,
        scheduler=None,
        lr=None,
        step_decay=None,
        learning_rate_warmup_steps=None,
        train_dataset=[],
        dev_dataset=[],
        test_dataset=[],
        batch_size=1,
        nprocs=1,
        elbo_objective=False,
        max_epochs=inf,
        max_time=inf,
        name='',
        checkpoint_dir='',
        output_dir='',
        data_dir='',
        evalb_dir='',
        log_dir='',
        max_grad_norm=5.0,
        print_every=100,
        device=None,
        args=None,  # Used for saving model.
    ):

        self.args = args
        self.model = model
        self.dictionary = model.dictionary
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.device = device

        self.elbo_objective = elbo_objective
        self.lr = lr
        self.step_decay = step_decay
        self.learning_rate_warmup_steps = learning_rate_warmup_steps
        self.max_grad_norm = max_grad_norm

        self.nprocs = nprocs
        self.distributed = (nprocs > 1)
        self.print_every = print_every
        self.max_epochs = max_epochs
        self.max_time = max_time

        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.name = name
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.evalb_dir = evalb_dir
        self.construct_paths()

        self.current_dev_fscore = -inf
        self.best_dev_fscore = -inf
        self.best_dev_epoch = None
        self.test_fscore = None

        self.losses = []
        self.num_updates = 0

        self.timer = Timer()

        self.tensorboard_writer = SummaryWriter(log_dir)

    def construct_paths(self):
        self.checkpoint_path = os.path.join(self.checkpoint_dir, 'model.pt')

        self.dev_pred_path = os.path.join(self.output_dir, f'{self.name}.dev.pred.trees')
        self.dev_gold_path = os.path.join(self.data_dir, 'dev', f'{self.name}.dev.trees')
        self.dev_result_path = os.path.join(self.output_dir, f'{self.name}.dev.result')

        self.test_pred_path = os.path.join(self.output_dir, f'{self.name}.test.pred.trees')
        self.test_gold_path = os.path.join(self.data_dir, 'test', f'{self.name}.test.trees')
        self.test_result_path = os.path.join(self.output_dir, f'{self.name}.test.result')

    def train(self):
        """Train the model.

        At any point you can hit Ctrl + C to break out of training early.
        """
        try:
            print('Training')
            # No upper limit of epochs.
            for epoch in itertools.count(start=1):
                if epoch > self.max_epochs:
                    break

                self.current_epoch = epoch

                # Shuffle batches each epoch.
                np.random.shuffle(self.train_dataset)

                # Train one epoch.
                self.train_epoch()

                print('Evaluating fscore on development set...')
                self.check_dev()

                # Scheduler for learning rate.
                if self.step_decay:
                    if (self.num_updates // self.batch_size + 1) > self.learning_rate_warmup_steps:
                        self.scheduler.step(self.best_dev_fscore)

                print('-'*99)
                print(
                    f'| End of epoch {epoch:3d}/{self.max_epochs} '
                    f'| elapsed {self.timer.format_elapsed()} '
                    f'| dev-fscore {self.current_dev_fscore:4.2f} '
                    f'| best dev-epoch {self.best_dev_epoch} '
                    f'| best dev-fscore {self.best_dev_fscore:4.2f} ',
                )
                print('-'*99)
        except KeyboardInterrupt:
            print('-'*99)
            print('Exiting from training early.')
            # Save the losses for plotting and diagnostics
            self.save_checkpoint()
            self.write_losses()
            print('Evaluating fscore on development set...')
            self.check_dev()

    def train_epoch(self):
        """One epoch of training."""
        self.model.train()
        train_timer = Timer()
        num_sentences = len(self.train_dataset)
        num_batches = num_sentences // self.batch_size
        processed = 0

        # if args.memory_debug:
        #     prev_mem = 0
        #     prev_num_objects, prev_num_tensors, prev_num_strings = 0, 0, 0
        #     old_tensors = []

        batches = self.batchify(self.train_dataset)
        for step, minibatch in enumerate(batches, 1):
            if train_timer.elapsed() > self.max_time:
                break

            # Set learning rate.
            self.num_updates += 1
            processed += self.batch_size
            self.schedule_lr()

            # Compute loss over minibatch.
            loss = torch.zeros(1).to(self.device)
            for batch in minibatch:
                sentence, actions = batch
                loss += self.model(sentence, actions)
            loss /= self.batch_size

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            self.losses.append(loss.item())

            ##
            if torch.isnan(loss.data):
                with open('checkpoints/nan-model.pt', 'w') as f:
                    torch.save(self.model, f)
                for param in self.model.parameters():
                    if torch.isnan(param.data).sum() > 0:
                        print(param)
                        print()
            ##

            if step % self.print_every == 0:
                # Log to tensorboard.
                self.tensorboard_writer.add_scalar(
                    'Train/Loss', loss.item(), self.num_updates)
                self.tensorboard_writer.add_scalar(
                    'Train/Learning-rate', self.get_lr(), self.num_updates)
                percentage = step / num_batches * 100
                avg_loss = np.mean(self.losses[-self.print_every:])
                lr = self.get_lr()
                sents_per_sec = processed / train_timer.elapsed()
                eta = (num_sentences - processed) / sents_per_sec

                message = (
                    f'| step {step:6d}/{num_batches:5d} ({percentage:.0f}%) ',
                    f'| loss {avg_loss:7.3f} ',
                    f'| lr {lr:.1e} ',
                    f'| {sents_per_sec:4.1f} sents/sec ',
                    f'| elapsed {train_timer.format(train_timer.elapsed())} ',
                    f'| eta {train_timer.format(eta)} '
                )
                if self.elbo_objective:
                    message += (
                        f'| alpha {self.model.criterion.annealer._alpha:.3f} ',
                        f'| temp {self.model.stack.encoder.composition.annealer._temp:.3f} '
                    )
                print(''.join(message))

    def schedule_lr(self):
        warmup_coeff = self.lr / self.learning_rate_warmup_steps
        if self.num_updates <= self.learning_rate_warmup_steps:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.num_updates * warmup_coeff

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def batchify(self, data):
        def ceil_div(a, b):
            return ((a - 1) // b) + 1

        batches = [data[i*self.batch_size:(i+1)*self.batch_size]
            for i in range(ceil_div(len(data), self.batch_size))]
        return batches

    def save_checkpoint(self):
        with open(self.checkpoint_path, 'wb') as f:
            state = {
                'args': self.args,
                'model': self.model,
                'dictionary': self.dictionary,
                'optimizer': self.optimizer,
                'scheduler': self.scheduler,
                'epochs': self.current_epoch,
                'num-updates': self.num_updates,
                'best-dev-fscore': self.best_dev_fscore,
                'best-dev-epoch': self.best_dev_epoch,
                'test-fscore': self.test_fscore
            }
            torch.save(state, f)

    def predict(self, batches):
        """Predict trees for the batches with the current model."""
        decoder = GreedyDecoder(
            model=self.model, dictionary=self.dictionary, use_chars=self.dictionary.use_chars)
        trees = []
        for i, batch in enumerate(batches):
            sentence, actions = batch
            tree, *rest = decoder(sentence)
            trees.append(tree)
            if i % 10 == 0:
                print(f'Predicting sentence {i}/{len(batches)}...', end='\r')
        return trees

    def check_dev(self):
        """Evaluate the current model on the test dataset."""
        self.model.eval()
        # Predict trees.
        trees = self.predict(self.dev_dataset)
        with open(self.dev_pred_path, 'w') as f:
            print('\n'.join([tree.linearize() for tree in trees]), file=f)
        # Compute f-score.
        dev_fscore = evalb(
            self.evalb_dir, self.dev_pred_path, self.dev_gold_path, self.dev_result_path)
        # Log score to tensorboard.
        self.tensorboard_writer.add_scalar('Dev/Fscore', dev_fscore, self.num_updates)
        self.current_dev_fscore = dev_fscore
        if dev_fscore > self.best_dev_fscore:
            print(f'Saving new best model to `{self.checkpoint_path}`...')
            self.best_dev_epoch = self.current_epoch
            self.best_dev_fscore = dev_fscore
            self.save_checkpoint()

    def check_test(self):
        """Evaluate the model with best development f-score on the test dataset."""
        print(f'Loading best saved model from `{self.checkpoint_path}` '
              f'(epoch {self.best_dev_epoch}, fscore {self.best_dev_fscore})...')
        with open(self.checkpoint_path, 'rb') as f:
            state = torch.load(f)
            self.model = state['model']
        # Predict trees.
        trees = self.predict(self.test_dataset)
        with open(self.test_pred_path, 'w') as f:
            print('\n'.join([tree.linearize() for tree in trees]), file=f)
        # Compute f-score.
        self.test_fscore = evalb(
            self.evalb_dir, self.test_pred_path, self.test_gold_path, self.test_result_path)
        print('-'*99)
        print(
             f'| End of training '
             f'| best dev-epoch {self.best_dev_epoch:2d} '
             f'| best dev-fscore {self.best_dev_fscore:4.2f} '
             f'| test-fscore {self.test_fscore}'
        )
        print('-'*99)
        # Save with test fscore information.
        self.save_checkpoint()

    def write_losses(self):
        path = os.path.join(self.log_dir, 'loss.csv')
        with open(path, 'w') as f:
            print('loss', file=f)
            for loss in self.losses:
                print(loss, file=f)
