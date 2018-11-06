import os
import json
import itertools
import time
import pickle
from math import inf

import numpy as np
import dynet as dy
from tensorboardX import SummaryWriter

from data import Corpus
from decode import GreedyDecoder, GenerativeImportanceDecoder
from model import DiscRNNG, GenRNNG
from eval import evalb
from utils import Timer, get_folders, write_args, ceil_div


class Trainer:
    """Trainer for RNNG."""
    def __init__(
            self,
            rnng_type='disc',
            args=None,
            name=None,
            data_dir=None,
            train_path=None,
            dev_path=None,
            test_path=None,
            text_type='unked',
            evalb_dir=None,
            dev_proposal_samples=None,
            test_proposal_samples=None,
            word_emb_dim=None,
            nt_emb_dim=None,
            action_emb_dim=None,
            stack_lstm_dim=None,
            buffer_lstm_dim=None,
            terminal_lstm_dim=None,
            history_lstm_dim=None,
            lstm_layers=None,
            composition=None,
            f_hidden_dim=None,
            max_epochs=inf,
            max_time=inf,
            lr=None,
            batch_size=1,
            dropout=0.,
            weight_decay=None,
            lr_decay=None,
            lr_decay_patience=None,
            max_grad_norm=None,
            use_glove=False,
            glove_dir=None,
            fine_tune_embeddings=False,
            freeze_embeddings=False,
            print_every=1,
            eval_every=-1,  # default is every epoch (-1)
            elbo_objective=False
    ):
        assert rnng_type in ('disc', 'gen'), rnng_type

        self.args = args

        # Data arguments
        self.name = name
        self.data_dir = data_dir
        self.evalb_dir = evalb_dir
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.text_type = text_type
        self.dev_proposal_samples = dev_proposal_samples
        self.test_proposal_samples = test_proposal_samples

        # Model arguments
        self.rnng_type = rnng_type
        self.word_emb_dim = word_emb_dim
        self.nt_emb_dim = nt_emb_dim
        self.action_emb_dim = action_emb_dim
        self.stack_lstm_dim = stack_lstm_dim
        self.buffer_lstm_dim = buffer_lstm_dim
        self.terminal_lstm_dim = terminal_lstm_dim
        self.history_lstm_dim = history_lstm_dim
        self.lstm_layers = lstm_layers
        self.composition = composition
        self.f_hidden_dim = f_hidden_dim
        self.dropout = dropout

        # Training arguments
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_patience = lr_decay_patience
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.elbo_objective = elbo_objective
        self.use_glove = use_glove
        self.glove_dir = glove_dir
        self.fine_tune_embeddings = fine_tune_embeddings
        self.freeze_embeddings = freeze_embeddings
        self.max_epochs = max_epochs
        self.max_time = max_time
        self.eval_every = eval_every
        self.print_every = print_every

        # Training bookkeeping
        self.losses = []
        self.num_updates = 0
        self.current_dev_fscore = -inf
        self.best_dev_fscore = -inf
        self.best_dev_epoch = 0
        self.test_fscore = -inf
        self.timer = Timer()

    def build_paths(self):
        # Make output folder structure
        subdir, logdir, checkdir, outdir = get_folders(self.args)  # TODO: make more transparent
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(checkdir, exist_ok=True)
        os.makedirs(outdir, exist_ok=True)
        print(f'Output subdirectory: `{subdir}`.')
        print(f'Saving logs to `{logdir}`.')
        print(f'Saving predictions to `{outdir}`.')
        print(f'Saving models to `{checkdir}`.')
        # Save arguments
        write_args(self.args, logdir)
        # Output paths
        self.model_checkpoint_path = os.path.join(checkdir, 'model')
        self.state_checkpoint_path = os.path.join(checkdir, 'state.json')
        self.dict_checkpoint_path = os.path.join(checkdir, 'dict.json')
        self.loss_path = os.path.join(logdir, 'loss.csv')
        self.tensorboard_writer = SummaryWriter(logdir)
        # Dev paths
        self.dev_gold_path = os.path.join(self.data_dir, 'dev', f'{self.name}.dev.trees')
        self.dev_pred_path = os.path.join(outdir, f'{self.name}.dev.pred.trees')
        self.dev_result_path = os.path.join(outdir, f'{self.name}.dev.result')
        # Test paths
        self.test_gold_path = os.path.join(self.data_dir, 'test', f'{self.name}.test.trees')
        self.test_pred_path = os.path.join(outdir, f'{self.name}.test.pred.trees')
        self.test_result_path = os.path.join(outdir, f'{self.name}.test.result')

    def build_corpus(self):
        # Get data
        corpus = Corpus(
            train_path=self.train_path,
            dev_path=self.dev_path,
            test_path=self.test_path,
            text_type=self.text_type,
            rnng_type=self.rnng_type
        )
        self.dictionary = corpus.dictionary
        self.train_dataset = corpus.train.data
        self.dev_dataset = corpus.dev.data
        self.test_dataset = corpus.test.data

    def build_model(self):
        assert self.dictionary is not None, 'build corpus first'

        self.model = dy.ParameterCollection()
        if self.rnng_type == 'disc':
            self.rnng = DiscRNNG(
                model=self.model,
                dictionary=self.dictionary,
                num_words=self.dictionary.num_words,
                num_nt=self.dictionary.num_nt,
                word_emb_dim=self.word_emb_dim,
                nt_emb_dim=self.nt_emb_dim,
                action_emb_dim=self.action_emb_dim,
                stack_lstm_dim=self.stack_lstm_dim,
                buffer_lstm_dim=self.buffer_lstm_dim,
                history_lstm_dim=self.history_lstm_dim,
                lstm_layers=self.lstm_layers,
                composition=self.composition,
                f_hidden_dim=self.f_hidden_dim,
                dropout=self.dropout,
                use_glove=self.use_glove,
                glove_dir=self.glove_dir,
                fine_tune_embeddings=self.fine_tune_embeddings,
                freeze_embeddings=self.freeze_embeddings,
            )
        elif self.rnng_type == 'gen':
            self.rnng = GenRNNG(
                model=self.model,
                dictionary=self.dictionary,
                num_words=self.dictionary.num_words,
                num_nt=self.dictionary.num_nt,
                word_emb_dim=self.word_emb_dim,
                nt_emb_dim=self.nt_emb_dim,
                action_emb_dim=self.action_emb_dim,
                stack_lstm_dim=self.stack_lstm_dim,
                terminal_lstm_dim=self.terminal_lstm_dim,
                history_lstm_dim=self.history_lstm_dim,
                lstm_layers=self.lstm_layers,
                composition=self.composition,
                f_hidden_dim=self.f_hidden_dim,
                dropout=self.dropout,
                use_glove=self.use_glove,
                glove_dir=self.glove_dir,
                fine_tune_embeddings=self.freeze_embeddings,
                freeze_embeddings=self.freeze_embeddings,
            )

    def build_optimizer(self):
        assert self.model is not None, 'build model first'

        if self.args.optimizer == 'sgd':
            self.optimizer = dy.SimpleSGDTrainer(self.model, learning_rate=self.lr)
        elif self.args.optimizer == 'adam':
            self.optimizer = dy.AdamTrainer(self.model, alpha=self.lr)
        self.optimizer.set_clip_threshold(self.max_grad_norm)
        self.model.set_weight_decay(self.weight_decay)

    def train(self):
        """
        Train the model. At any point you can
        hit Ctrl + C to break out of training early.
        """
        if self.rnng_type == 'gen':
            # These are needed for evaluation
            assert self.dev_proposal_samples is not None, 'specify proposal samples with --dev-proposal-samples.'
            assert self.test_proposal_samples is not None, 'specify proposal samples with --test-proposal-samples.'
            assert os.path.exists(self.dev_proposal_samples), self.dev_proposal_samples
            assert os.path.exists(self.test_proposal_samples), self.test_proposal_samples
        # Construct model and optimizer
        self.build_paths()
        self.build_corpus()
        self.build_model()
        self.build_optimizer()
        # No upper limit of epochs or time when not specified
        for epoch in itertools.count(start=1):
            if epoch > self.max_epochs:
                break
            if self.timer.elapsed() > self.max_time:
                break
            self.current_epoch = epoch
            # Shuffle batches every epoch
            np.random.shuffle(self.train_dataset)
            # Train one epoch
            self.train_epoch()
            # Check development f-score
            self.check_dev()
            # Anneal learning rate depending on development set f-score
            self.anneal_lr()
            print('-'*99)
            print('| End of epoch {:3d}/{} | Elapsed {} | Current dev F1 {:4.2f} | Best dev F1 {:4.2f} (epoch {:2d})'.format(
                epoch, self.max_epochs, self.timer.format_elapsed(), self.current_dev_fscore, self.best_dev_fscore, self.best_dev_epoch))
            print('-'*99)

    def train_epoch(self):
        """One epoch of sequential training."""
        self.rnng.train()
        epoch_timer = Timer()
        num_sentences = len(self.train_dataset)
        num_batches = num_sentences // self.batch_size
        processed = 0
        batches = self.batchify(self.train_dataset)
        for step, minibatch in enumerate(batches, 1):
            if self.timer.elapsed() > self.max_time:
                break
            # Keep track of updates
            self.num_updates += 1
            processed += self.batch_size
            # Compute loss on minibatch
            dy.renew_cg()
            loss = dy.esum([self.rnng(words, actions) for words, actions in minibatch])
            loss /= self.batch_size
            # Add penalty if fine-tuning embeddings
            if self.fine_tune_embeddings:
                delta_penalty = self.rnng.word_embedding.delta_penalty()
                loss += delta_penalty
            # Update parameters
            loss.forward()
            loss.backward()
            self.optimizer.update()
            # Bookkeeping
            self.losses.append(loss.value())
            if step % self.print_every == 0:
                # Info for terminal.
                avg_loss = np.mean(self.losses[-self.print_every:])
                lr = self.get_lr()
                sents_per_sec = processed / epoch_timer.elapsed()
                updates_per_sec = self.num_updates / epoch_timer.elapsed()
                eta = (num_sentences - processed) / sents_per_sec
                # Log to tensorboard.
                self.tensorboard_writer.add_scalar(
                    'train/loss', avg_loss, self.num_updates)
                self.tensorboard_writer.add_scalar(
                    'train/learning-rate', self.get_lr(), self.num_updates)
                if self.fine_tune_embeddings:
                    self.tensorboard_writer.add_scalar(
                        'train/embedding-l2', delta_penalty.value(), self.num_updates)
                message = (
                    f'| step {step:6d}/{num_batches:5d} ({step/num_batches:.0%}) ',
                    f'| loss {avg_loss:7.3f} ',
                    f'| lr {lr:.1e} ',
                    f'| {sents_per_sec:4.1f} sents/sec ',
                    f'| {updates_per_sec:4.1f} updates/sec ',
                    f'| elapsed {epoch_timer.format(epoch_timer.elapsed())} ',
                    f'| eta {epoch_timer.format(eta)} '
                )
                print(''.join(message))

    def get_lr(self):
        return self.optimizer.learning_rate

    def set_lr(self, lr):
        self.optimizer.learning_rate = lr

    def batchify(self, data):
        batches = [data[i*self.batch_size:(i+1)*self.batch_size]
            for i in range(ceil_div(len(data), self.batch_size))]
        return batches

    def anneal_lr(self):
        if self.current_dev_fscore < self.best_dev_fscore:  # if F1 has gotten worse
            if self.current_epoch > (self.best_dev_epoch + self.lr_decay_patience):  # if we've waited long enough
                lr = self.get_lr() / self.lr_decay
                print(f'Annealing the learning rate from {self.get_lr():.1e} to {lr:.1e}.')
                self.set_lr(lr)

    def save_checkpoint(self):
        assert self.model is not None, 'no model built'

        dy.save(self.model_checkpoint_path, [self.rnng])
        self.dictionary.save(self.dict_checkpoint_path)
        with open(self.state_checkpoint_path, 'w') as f:
            state = {
                'rnng-type': self.rnng_type,
                'text-type': self.text_type,
                'epochs': self.current_epoch,
                'num-updates': self.num_updates,
                'best-dev-fscore': self.best_dev_fscore,
                'best-dev-epoch': self.best_dev_epoch,
                'test-fscore': self.test_fscore,
            }
            json.dump(state, f, indent=4)

    def load_checkpoint(self):
        self.model = dy.ParameterCollection()
        [self.rnng] = dy.load(self.model_checkpoint_path, self.model)
        with open(self.state_checkpoint_path) as f:
            state = json.load(f)

    def predict(self, examples, proposal_samples=None):
        if self.rnng_type == 'disc':
            decoder = GreedyDecoder(
                model=self.rnng, dictionary=self.dictionary)
        elif self.rnng_type == 'gen':
            assert proposal_samples is not None, 'path to samples required for generative decoding.'
            decoder = GenerativeImportanceDecoder(
                model=self.rnng, dictionary=self.dictionary)
            decoder.load_proposal_samples(path=proposal_samples)
        trees = []
        for i, (sentence, actions) in enumerate(examples):
            dy.renew_cg()
            tree, *rest = decoder(sentence)
            tree = tree if isinstance(tree, str) else tree.linearize()
            trees.append(tree)
            if i % 10 == 0:
                print(f'Predicting sentence {i}/{len(examples)}...', end='\r')
        return trees

    def check_dev(self):
        print('Evaluating F1 on development set...')
        self.rnng.eval()
        # Predict trees.
        dev_dataset = self.dev_dataset[:30] if self.rnng_type == 'gen' else self.dev_dataset
        trees = self.predict(dev_dataset, proposal_samples=self.dev_proposal_samples)
        with open(self.dev_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)
        # Compute f-score.
        dev_fscore = evalb(
            self.evalb_dir, self.dev_pred_path, self.dev_gold_path, self.dev_result_path)
        # Log score to tensorboard.
        self.tensorboard_writer.add_scalar('dev/f-score', dev_fscore, self.num_updates)
        self.current_dev_fscore = dev_fscore
        if dev_fscore > self.best_dev_fscore:
            print(f'Saving new best model to `{self.model_checkpoint_path}`...')
            self.best_dev_epoch = self.current_epoch
            self.best_dev_fscore = dev_fscore
            self.save_checkpoint()
        return dev_fscore

    def check_test(self):
        print('Evaluating F1 on test set...')
        print(f'Loading best saved model from `{self.model_checkpoint_path}` '
              f'(epoch {self.best_dev_epoch}, fscore {self.best_dev_fscore})...')
        self.load_checkpoint()
        self.rnng.eval()
        # Predict trees.
        trees = self.predict(self.test_dataset, proposal_samples=self.test_proposal_samples)
        with open(self.test_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)
        # Compute f-score.
        test_fscore = evalb(
            self.evalb_dir, self.test_pred_path, self.test_gold_path, self.test_result_path)
        self.tensorboard_writer.add_scalar('test/f-score', test_fscore)
        return test_fscore

    def write_losses(self):
        with open(self.loss_path, 'w') as f:
            print('loss', file=f)
            for loss in self.losses:
                print(loss, file=f)


class SemiSupervisedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SemiSupervisedTrainer, self).__init__(*args, **kwargs)
