import os
import json
import itertools
import time
import multiprocessing as mp
from math import inf

import numpy as np
import dynet as dy
from tensorboardX import SummaryWriter

from data import Corpus
# from decode import GreedyDecoder, GenerativeImportanceDecoder
from model import DiscRNNG, GenRNNG
from eval import evalb
from utils import Timer, get_folders, write_args, ceil_div


class Trainer:
    """Trainer for RNNG."""
    def __init__(self,
                 rnng_type='disc',
                 dictionary=None,
                 optimizer=None,
                 lr=None,
                 learning_rate_decay=2.0,  # lr /= learning_rate_decay
                 max_grad_norm=5.0,
                 weight_decay=1e-6,
                 use_glove=False,
                 glove_dir=None,
                 fine_tune_embeddings=False,
                 freeze_embeddings=False,
                 print_every=1,
                 eval_every=-1,  # default is every epoch (-1)
                 train_dataset=[],
                 dev_dataset=[],
                 test_dataset=[],
                 dev_proposal_samples=None,
                 test_proposal_samples=None,
                 batch_size=1,
                 elbo_objective=False,
                 max_epochs=inf,
                 max_time=inf,
                 name=None,
                 checkpoint_dir=None,
                 output_dir=None,
                 data_dir=None,
                 evalb_dir=None,
                 log_dir=None,
                 args=None):  # used when saving model
        assert rnng_type in ('disc', 'gen'), rnng_type

        self.rnng_type = rnng_type
        self.dictionary = dictionary
        self.args = args

        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.learning_rate_decay = learning_rate_decay
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.elbo_objective = elbo_objective
        self.use_glove = use_glove
        self.glove_dir = glove_dir
        self.fine_tune_embeddings = fine_tune_embeddings
        self.freeze_embeddings = freeze_embeddings

        self.print_every = print_every
        self.max_epochs = max_epochs
        self.max_time = max_time

        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.dev_proposal_samples = dev_proposal_samples
        self.test_proposal_samples = test_proposal_samples

        self.name = name
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.evalb_dir = evalb_dir
        self.construct_paths()

        self.losses = []
        self.num_updates = 0
        self.eval_every = eval_every
        self.current_dev_fscore = -inf
        self.best_dev_fscore = -inf
        self.best_dev_epoch = 0
        self.test_fscore = -inf

        self.timer = Timer()
        self.tensorboard_writer = SummaryWriter(log_dir)

    def construct_paths(self):
        # Output paths
        self.model_checkpoint_path = os.path.join(self.checkpoint_dir, 'model.dy')
        self.state_checkpoint_path = os.path.join(self.checkpoint_dir, 'state.json')
        self.dict_checkpoint_path = os.path.join(self.checkpoint_dir, 'dict.json')
        self.loss_path = os.path.join(self.log_dir, 'loss.csv')
        # Dev paths
        self.dev_gold_path = os.path.join(self.data_dir, 'dev', f'{self.name}.dev.trees')
        self.dev_pred_path = os.path.join(self.output_dir, f'{self.name}.dev.pred.trees')
        self.dev_result_path = os.path.join(self.output_dir, f'{self.name}.dev.result')
        # Test paths
        self.test_gold_path = os.path.join(self.data_dir, 'test', f'{self.name}.test.trees')
        self.test_pred_path = os.path.join(self.output_dir, f'{self.name}.test.pred.trees')
        self.test_result_path = os.path.join(self.output_dir, f'{self.name}.test.result')

    def train(self):
        """
        Train the model. At any point you can
        hit Ctrl + C to break out of training early.
        """
        if self.rnng_type == 'gen':
            # These are needed for evaluation.
            assert self.dev_proposal_samples is not None, 'specify proposal samples with --dev-proposal-samples.'
            assert self.test_proposal_samples is not None, 'specify proposal samples with --test-proposal-samples.'
            assert os.path.exists(self.dev_proposal_samples), self.dev_proposal_samples
            assert os.path.exists(self.test_proposal_samples), self.test_proposal_samples

        # Construct model and optimizer
        self.build_model()
        self.build_optimizer()

        # No upper limit of epochs or time when not specified
        for epoch in itertools.count(start=1):
            self.current_epoch = epoch
            if epoch > self.max_epochs:
                break
            if self.timer.elapsed() > self.max_time:
                break

            # Shuffle batches every epoch
            np.random.shuffle(self.train_dataset)

            # Train one epoch.
            self.train_epoch()

            if self.eval_every == -1:
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
        ##
        # self.rnng.train()
        ##
        epoch_timer = Timer()
        num_sentences = len(self.train_dataset)
        num_batches = num_sentences // self.batch_size
        processed = 0
        batches = self.batchify(self.train_dataset)
        for step, minibatch in enumerate(batches, 1):
            if self.timer.elapsed() > self.max_time:
                break

            self.num_updates += 1
            processed += self.batch_size

            dy.renew_cg()

            loss = dy.esum([self.rnng(words, actions) for words, actions in minibatch])
            loss /= self.batch_size

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

            if self.eval_every != -1 and self.num_updates % self.eval_every == 0:
                self.check_dev()
                print('Current development F1 {:4.2f} (best {:4.2f} epoch {:2d})'.format(
                    self.current_dev_fscore, self.best_dev_fscore, self.best_dev_epoch))
                self.anneal_lr()
                # self.rnng.train()  # back to training mode

    def get_lr(self):
        return self.optimizer.learning_rate

    def set_lr(self, lr):
        self.optimizer.learing_rate = lr

    def batchify(self, data):
        batches = [data[i*self.batch_size:(i+1)*self.batch_size]
            for i in range(ceil_div(len(data), self.batch_size))]
        return batches

    def anneal_lr(self):
        if self.current_dev_fscore < self.best_dev_fscore:  # if current F1 is worse
            lr = self.get_lr() / self.learning_rate_decay
            print(f'Annealing the learning rate from {self.get_lr():.1e} to {lr:.1e}.')
            self.set_lr(lr)

    def build_model(self):
        model = dy.Model()
        if self.rnng_type == 'disc':
            rnng = DiscRNNG(
                model=model,
                dictionary=self.dictionary,
                num_words=len(self.dictionary.w2i),
                num_nt=len(self.dictionary.n2i),
                word_emb_dim=self.args.word_emb_dim,
                nt_emb_dim=self.args.nt_emb_dim,
                action_emb_dim=self.args.action_emb_dim,
                stack_hidden_size=self.args.stack_lstm_hidden,
                buffer_hidden_size=self.args.buffer_lstm_hidden,
                history_hidden_size=self.args.history_lstm_hidden,
                stack_num_layers=self.args.lstm_num_layers,
                buffer_num_layers=self.args.lstm_num_layers,
                history_num_layers=self.args.lstm_num_layers,
                composition=self.args.composition,
                mlp_hidden=self.args.mlp_hidden,
                dropout=self.args.dropout,
                use_glove=self.use_glove,
                glove_dir=self.glove_dir,
                fine_tune_embeddings=self.fine_tune_embeddings,
                freeze_embeddings=self.freeze_embeddings,
            )
        elif self.rnng_type == 'gen':
            rnng = GenRNNG(
                model=model,
                dictionary=self.dictionary,
                num_words=len(self.dictionary.w2i),
                num_nt=len(self.dictionary.n2i),
                word_emb_dim=self.args.word_emb_dim,
                nt_emb_dim=self.args.nt_emb_dim,
                action_emb_dim=self.args.action_emb_dim,
                stack_hidden_size=self.args.stack_lstm_hidden,
                terminal_hidden_size=self.args.terminal_lstm_hidden,
                history_hidden_size=self.args.history_lstm_hidden,
                stack_num_layers=self.args.lstm_num_layers,
                terminal_num_layers=self.args.lstm_num_layers,
                history_num_layers=self.args.lstm_num_layers,
                composition=self.args.composition,
                mlp_hidden=self.args.mlp_hidden,
                dropout=self.args.dropout,
                use_glove=self.use_glove,
                glove_dir=self.glove_dir,
                fine_tune_embeddings=self.freeze_embeddings,
                freeze_embeddings=self.freeze_embeddings,
            )
        self.model = model
        self.rnng = rnng

    def build_optimizer(self):
        assert self.model is not None, 'build model first'
        if self.args.optimizer == 'sgd':
            self.optimizer = dy.SimpleSGDTrainer(self.model, learning_rate=self.lr)
        elif self.args.optimizer == 'adam':
            self.optimizer = dy.AdamTrainer(self.model, alpha=self.lr)
        self.optimizer.set_clip_threshold(self.max_grad_norm)
        self.model.set_weight_decay(self.weight_decay)

    def save_checkpoint(self):
        assert self.model is not None, 'build model first'
        self.model.save(self.model_checkpoint_path)
        self.dictionary.save(self.dict_checkpoint_path)
        with open(self.state_checkpoint_path, 'w') as f:
            state = {
                'epochs': self.current_epoch,
                'num-updates': self.num_updates,
                'best-dev-fscore': self.best_dev_fscore,
                'best-dev-epoch': self.best_dev_epoch,
                'test-fscore': self.test_fscore,
            }
            json.dump(state, f, indent=4)

    def load_checkpoint(self):
        if self.model is None:
            self.build_model()
            self.build_optimizer()
        self.model.populate(self.model_checkpoint_path)

    ## TODO
    def _predict(self, batches, proposal_samples=None):
        if self.rnng_type == 'disc':
            decoder = GreedyDecoder(
                model=self.rnng, dictionary=self.dictionary, use_chars=self.dictionary.use_chars)
        elif self.rnng_type == 'gen':
            assert proposal_samples is not None, 'path to samples required for generative decoding.'
            decoder = GenerativeImportanceDecoder(
                model=self.rnng, dictionary=self.dictionary, use_chars=self.dictionary.use_chars)
            decoder.load_proposal_samples(path=proposal_samples)
            batches = batches[:100]  # Otherwise this will take 50 minutes.
        trees = []
        for i, batch in enumerate(batches):
            sentence, actions = batch
            tree, *rest = decoder(sentence)
            tree = tree if isinstance(tree, str) else tree.linearize()
            trees.append(tree)
            if i % 10 == 0:
                print(f'Predicting sentence {i}/{len(batches)}...', end='\r')
        return trees

    def predict(self, batches, proposal_samples=None):
        trees = []
        for i, batch in enumerate(batches):
            sentence, actions = batch
            tree, nll = self.rnng.parse(sentence)
            nll.forward()
            trees.append(tree.linearize())
            if i % 10 == 0:
                print(f'Predicting sentence {i}/{len(batches)}...', end='\r')
        return trees

    def check_dev(self):
        print('Evaluating F1 on development set...')
        ## TODO:
        # self.rnng.eval()
        ##
        # Predict trees.
        trees = self.predict(self.dev_dataset, proposal_samples=self.dev_proposal_samples)
        with open(self.dev_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)
        # Compute f-score.
        dev_fscore = evalb(
            self.evalb_dir, self.dev_pred_path, self.dev_gold_path, self.dev_result_path)
        # Log score to tensorboard.
        self.tensorboard_writer.add_scalar('Dev/Fscore', dev_fscore, self.num_updates)
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
        ## TODO:
        # self.rnng.eval()
        ##
        # Predict trees.
        trees = self.predict(self.test_dataset, proposal_samples=self.test_proposal_samples)
        with open(self.test_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)
        # Compute f-score.
        test_fscore = evalb(
            self.evalb_dir, self.test_pred_path, self.test_gold_path, self.test_result_path)
        return test_fscore

    def write_losses(self):
        with open(self.loss_path, 'w') as f:
            print('loss', file=f)
            for loss in self.losses:
                print(loss, file=f)


class SemiSupervisedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SemiSupervisedTrainer, self).__init__(*args, **kwargs)
