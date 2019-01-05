import os
import json
import itertools
from math import inf
from collections import Counter

import numpy as np
import dynet as dy
from tqdm import tqdm
from tensorboardX import SummaryWriter

from lm.model import LanguageModel, MultitaskLanguageModel, START, STOP
from utils.vocabulary import Vocabulary, UNK
from utils.trees import fromstring, DUMMY
from utils.text import replace_quotes, replace_brackets
from utils.general import Timer, get_folders, write_args, ceil_div, move_to_final_folder


class LanguageModelTrainer:
    """Supervised trainer for RNNG."""
    def __init__(
            self,
            model_path_base=None,
            multitask=False,
            predict_all_spans=False,
            args=None,
            train_path=None,
            dev_path=None,
            test_path=None,
            vocab_path=None,
            emb_dim=None,
            lstm_dim=None,
            lstm_layers=None,
            label_hidden_dim=None,
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
    ):
        self.args = args

        # Data arguments
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.vocab_path = vocab_path

        # Model arguments
        self.model_path_base = model_path_base
        self.multitask = multitask
        self.predict_all_spans = predict_all_spans
        self.emb_dim = emb_dim
        self.lstm_dim = lstm_dim
        self.lstm_layers = lstm_layers
        self.label_hidden_dim = label_hidden_dim
        self.dropout = dropout

        # Training arguments
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_patience = lr_decay_patience
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.use_glove = use_glove
        self.glove_dir = glove_dir
        self.fine_tune_embeddings = fine_tune_embeddings
        self.freeze_embeddings = freeze_embeddings
        self.max_epochs = max_epochs
        self.max_time = max_time
        self.eval_every = eval_every
        self.print_every = print_every

        # Training bookkeeping
        self.timer = Timer()
        self.losses = []
        self.num_updates = 0

        self.current_dev_pp = inf
        self.best_dev_pp = inf
        self.best_dev_epoch = 0
        self.test_pp = inf

    def build_paths(self):
        # Make output folder structure
        subdir, logdir, checkdir, outdir, vocabdir = get_folders(self.args)
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(checkdir, exist_ok=True)
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(vocabdir, exist_ok=True)
        print(f'Output subdirectory: `{subdir}`.')
        print(f'Saving logs to `{logdir}`.')
        print(f'Saving predictions to `{outdir}`.')
        print(f'Saving models to `{checkdir}`.')

        # Save arguments
        write_args(self.args, logdir)

        # Output paths
        self.subdir = subdir
        self.model_checkpoint_path = os.path.join(checkdir, 'model')
        self.state_checkpoint_path = os.path.join(checkdir, 'state.json')
        self.word_vocab_path = os.path.join(vocabdir, 'word-vocab.json')
        self.label_vocab_path = os.path.join(vocabdir, 'nt-vocab.json')
        self.loss_path = os.path.join(logdir, 'loss.csv')
        self.tensorboard_writer = SummaryWriter(logdir)

    def build_corpus(self):
        print(f'Loading training trees from `{self.train_path}`...')
        with open(self.train_path) as f:
            train_treebank = [fromstring(line.strip()) for line in f]

        print(f'Loading development trees from `{self.dev_path}`...')
        with open(self.dev_path) as f:
            dev_treebank = [fromstring(line.strip()) for line in f]

        print(f'Loading test trees from `{self.test_path}`...')
        with open(self.test_path) as f:
            test_treebank = [fromstring(line.strip()) for line in f]

        if self.multitask:
            # need trees with span-information
            train_treebank = [tree.convert() for tree in train_treebank]
            dev_treebank = [tree.convert() for tree in dev_treebank]
            test_treebank = [tree.convert() for tree in test_treebank]

        print("Constructing vocabularies...")
        if self.vocab_path is not None:
            print(f'Using word vocabulary specified in `{self.vocab_path}`')
            with open(self.vocab_path) as f:
                vocab = json.load(f)
            words = [word for word, count in vocab.items() for _ in range(count)]
        else:
            words = [word for tree in train_treebank for word in tree.words()]

        if self.multitask:
            labels = [label for tree in train_treebank for label in tree.labels()]
        else:
            labels = []

        if self.multitask:
            words = [UNK, START, STOP] + words
        else:
            words = [UNK, START] + words

        word_vocab = Vocabulary.fromlist(words, unk_value=UNK)
        label_vocab = Vocabulary.fromlist(labels)

        self.word_vocab = word_vocab
        self.label_vocab = label_vocab

        self.train_treebank = train_treebank
        self.dev_treebank = dev_treebank
        self.test_treebank = test_treebank

        print('\n'.join((
            'Corpus statistics:',
            f'Vocab: {word_vocab.size:,} words, {label_vocab.size:,} nonterminals',
            f'Train: {len(train_treebank):,} sentences',
            f'Dev: {len(dev_treebank):,} sentences',
            f'Test: {len(test_treebank):,} sentences')))

    def build_model(self):
        assert self.word_vocab is not None, 'build corpus first'

        print('Initializing model...')
        self.model = dy.ParameterCollection()

        if self.multitask:
            lm = MultitaskLanguageModel(
                model=self.model,
                word_vocab=self.word_vocab,
                label_vocab=self.label_vocab,
                word_embedding_dim=self.emb_dim,
                lstm_dim=self.lstm_dim,
                lstm_layers=self.lstm_layers,
                label_hidden_dim=self.label_hidden_dim,
                dropout=self.dropout,
                predict_all_spans=self.predict_all_spans
            )
        else:
            lm = LanguageModel(
                model=self.model,
                word_vocab=self.word_vocab,
                word_embedding_dim=self.emb_dim,
                lstm_dim=self.lstm_dim,
                lstm_layers=self.lstm_layers,
                dropout=self.dropout
            )
        self.lm = lm
        print('Number of parameters: {:,}'.format(self.lm.num_params))

    def build_optimizer(self):
        assert self.model is not None, 'build model first'

        print(f'Building {self.args.optimizer} optimizer...')
        if self.args.optimizer == 'sgd':
            self.optimizer = dy.SimpleSGDTrainer(self.model, learning_rate=self.lr)
        elif self.args.optimizer == 'adam':
            self.optimizer = dy.AdamTrainer(self.model, alpha=self.lr)

        self.optimizer.set_clip_threshold(self.max_grad_norm)
        self.model.set_weight_decay(self.weight_decay)

    def train(self):
        """
        Train the model. At any point you can
        use Ctrl + C to break out of training early.
        """

        # Construct model and optimizer
        self.build_paths()
        self.build_corpus()
        self.build_model()
        self.build_optimizer()

        try:
            # No upper limit of epochs or time when not specified
            print('Start training...')
            for epoch in itertools.count(start=1):
                if epoch > self.max_epochs:
                    break
                if self.timer.elapsed() > self.max_time:
                    break
                self.current_epoch = epoch

                # Shuffle batches every epoch
                np.random.shuffle(self.train_treebank)

                # Train one epoch
                self.train_epoch()

                # Check development scores
                self.check_dev()

                # Anneal learning rate depending on development set f-score
                self.anneal_lr()

                print('-'*99)
                print('| End of epoch {:3d}/{} | Elapsed {} | Current dev pp {:4.2f} | Best dev pp {:4.2f} (epoch {:2d})'.format(
                    epoch, self.max_epochs, self.timer.format_elapsed(), self.current_dev_pp, self.best_dev_pp, self.best_dev_epoch))
                print('-'*99)
        except KeyboardInterrupt:
            print('-'*99)
            print('Exiting from training early.')
            print('-'*99)

        self.check_dev()

        # Check test scores
        self.check_test()

        # Save model again but with test pp
        self.save_checkpoint()

        # Save the losses for plotting and diagnostics
        self.write_losses()

        print('='*99)
        print('| End of training | Best dev pp {:3.2f} (epoch {:2d}) | Test pp {:3.2f}'.format(
            self.best_dev_pp, self.best_dev_epoch, self.test_pp))
        print('='*99)

    def train_epoch(self):
        """One epoch of sequential training."""
        self.lm.train()
        epoch_timer = Timer()
        num_sentences = len(self.train_treebank)
        num_batches = num_sentences // self.batch_size
        processed = 0
        batches = self.batchify(self.train_treebank)
        for step, minibatch in enumerate(batches, 1):
            if self.timer.elapsed() > self.max_time:
                break

            # Keep track of updates
            self.num_updates += 1
            processed += self.batch_size

            # Compute loss on minibatch
            dy.renew_cg()
            if self.multitask:
                losses = [self.lm.forward(tree.words(), spans=tree.spans()) for tree in minibatch]
            else:
                losses = [self.lm.forward(tree.words()) for tree in minibatch]
            loss = dy.esum(losses)
            loss /= self.batch_size

            # Add penalty if fine-tuning embeddings
            if self.fine_tune_embeddings:
                delta_penalty = self.lm.word_embedding.delta_penalty()
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

                print('| epoch {} | step {:6d}/{:5d} ({:.0%}) | loss {:7.3f} | lr {:.1e} | {:4.1f} sents/sec | {:4.1f} updates/sec | elapsed {} | eta {} '.format(
                    self.current_epoch, step, num_batches, step/num_batches, avg_loss, lr, sents_per_sec, updates_per_sec,
                    epoch_timer.format(epoch_timer.elapsed()), epoch_timer.format(eta)))

                # Info for tensorboard.
                self.tensorboard_writer.add_scalar(
                    'train/loss', avg_loss, self.num_updates)
                self.tensorboard_writer.add_scalar(
                    'train/learning-rate', self.get_lr(), self.num_updates)
                if self.fine_tune_embeddings:
                    self.tensorboard_writer.add_scalar(
                        'train/embedding-l2', delta_penalty.value(), self.num_updates)
                if self.multitask:
                    self.tensorboard_writer.add_scalar(
                        'train/scaffold-accuracy', self.lm.correct / self.lm.predicted, self.num_updates)

    def get_lr(self):
        return self.optimizer.learning_rate

    def set_lr(self, lr):
        self.optimizer.learning_rate = lr

    def batchify(self, data):
        batches = [data[i*self.batch_size:(i+1)*self.batch_size]
            for i in range(ceil_div(len(data), self.batch_size))]
        return batches

    def anneal_lr(self):
        if self.current_dev_pp > self.best_dev_pp:
            if self.current_epoch > (self.best_dev_epoch + self.lr_decay_patience):  # if we've waited long enough
                lr = self.get_lr() / self.lr_decay
                print(f'Annealing the learning rate from {self.get_lr():.1e} to {lr:.1e}.')
                self.set_lr(lr)

    def save_checkpoint(self):
        assert self.model is not None, 'no model built'

        dy.save(self.model_checkpoint_path, [self.lm])

        self.word_vocab.save(self.word_vocab_path)
        self.label_vocab.save(self.label_vocab_path)

        with open(self.state_checkpoint_path, 'w') as f:
            state = {
                'model': 'rnn-lm',
                'multitask': self.multitask,
                'num-params': int(self.lm.num_params),
                'num-epochs': self.current_epoch,
                'num-updates': self.num_updates,
                'current-lr': self.get_lr(),
                'best-dev-pp': self.best_dev_pp,
                'best-dev-epoch': self.best_dev_epoch,
                'test-pp': self.test_pp,
            }
            json.dump(state, f, indent=4)

    def load_checkpoint(self):
        self.model = dy.ParameterCollection()
        [self.lm] = dy.load(self.model_checkpoint_path, self.model)

    def perplexity(self, treebank):
        nll = 0
        num_words = 0
        self.lm.eval()
        for tree in tqdm(treebank):
            dy.renew_cg()
            words = tree.words()
            num_words += len(words)
            nll += self.lm.forward(words).value()
        self.lm.train()
        pp = np.exp(nll / num_words)
        return round(pp, 2)

    def check_dev(self):
        print('Evaluating perplexity on development set...')

        dev_pp = self.perplexity(self.dev_treebank)

        # Log score to tensorboard
        self.tensorboard_writer.add_scalar(
            'dev/pp', dev_pp, self.num_updates)

        self.current_dev_pp = dev_pp
        if dev_pp < self.best_dev_pp:
            print(f'Saving new best model to `{self.model_checkpoint_path}`...')
            self.best_dev_epoch = self.current_epoch
            self.best_dev_pp = dev_pp
            self.save_checkpoint()

    def check_test(self):
        print('Evaluating perplexity on test set...')

        test_pp = self.perplexity(self.test_treebank)

        # Log score to tensorboard
        self.tensorboard_writer.add_scalar(
            'test/pp', test_pp, self.num_updates)
        self.test_pp = test_pp

    def write_losses(self):
        with open(self.loss_path, 'w') as f:
            print('loss', file=f)
            for loss in self.losses:
                print(loss, file=f)

    def finalize_model_folder(self):
        move_to_final_folder(
            self.subdir, self.model_path_base, self.best_dev_pp)
