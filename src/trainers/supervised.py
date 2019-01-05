import os
import json
import itertools
from math import inf
from collections import Counter

import numpy as np
import dynet as dy
from tqdm import tqdm
from tensorboardX import SummaryWriter

from rnng.parser.actions import SHIFT, REDUCE, NT, GEN
from rnng.model import DiscRNNG, GenRNNG
from rnng.decoder import GenerativeDecoder
from crf.model import ChartParser, START, STOP
from utils.vocabulary import Vocabulary, UNK
from utils.trees import fromstring, DUMMY
from utils.evalb import evalb
from utils.text import replace_quotes, replace_brackets
from utils.general import Timer, get_folders, write_args, ceil_div, move_to_final_folder


class SupervisedTrainer:
    """Supervised trainer for RNNG."""
    def __init__(
            self,
            parser_type=None,
            model_path_base=None,
            args=None,
            train_path=None,
            dev_path=None,
            test_path=None,
            vocab_path=None,
            evalb_dir=None,
            dev_proposal_samples=None,
            test_proposal_samples=None,
            word_emb_dim=None,
            label_emb_dim=None,
            action_emb_dim=None,
            stack_lstm_dim=None,
            buffer_lstm_dim=None,
            terminal_lstm_dim=None,
            history_lstm_dim=None,
            lstm_dim=None,
            lstm_layers=None,
            composition=None,
            f_hidden_dim=None,
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
        assert parser_type in ('disc-rnng', 'gen-rnng', 'crf'), parser_type

        self.args = args

        # Data arguments
        self.evalb_dir = evalb_dir
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.vocab_path = vocab_path
        self.dev_proposal_samples = dev_proposal_samples
        self.test_proposal_samples = test_proposal_samples

        # Model arguments
        self.parser_type = parser_type
        self.model_path_base = model_path_base
        self.word_emb_dim = word_emb_dim
        self.label_emb_dim = label_emb_dim
        self.action_emb_dim = action_emb_dim
        self.stack_lstm_dim = stack_lstm_dim
        self.buffer_lstm_dim = buffer_lstm_dim
        self.terminal_lstm_dim = terminal_lstm_dim
        self.history_lstm_dim = history_lstm_dim
        self.lstm_dim = lstm_dim
        self.lstm_layers = lstm_layers
        self.composition = composition
        self.f_hidden_dim = f_hidden_dim
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

        self.current_dev_fscore = -inf
        self.best_dev_fscore = -inf
        self.best_dev_epoch = 0
        self.test_fscore = -inf

        self.current_dev_perplexity = -inf
        self.best_dev_perplexity = -inf
        self.best_dev_perplexity_epoch = 0
        self.test_perplexity = -inf

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
        self.label_vocab_path = os.path.join(vocabdir, 'label-vocab.json')
        self.action_vocab_path = os.path.join(vocabdir, 'action-vocab.json')
        self.loss_path = os.path.join(logdir, 'loss.csv')
        self.tensorboard_writer = SummaryWriter(logdir)

        # Dev paths
        self.dev_pred_path = os.path.join(outdir, 'dev.pred.trees')
        self.dev_result_path = os.path.join(outdir, 'dev.result')

        # Test paths
        self.test_pred_path = os.path.join(outdir, 'test.pred.trees')
        self.test_result_path = os.path.join(outdir, 'test.result')

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

        if self.parser_type == 'crf':
            print(f'Converting trees to CNF...')
            train_treebank = [tree.cnf() for tree in train_treebank]
            dev_treebank = [tree.cnf() for tree in dev_treebank]
            test_treebank = [tree.cnf() for tree in test_treebank]

        print("Constructing vocabularies...")
        if self.vocab_path is not None:
            print(f'Using word vocabulary specified in `{self.vocab_path}`')
            with open(self.vocab_path) as f:
                vocab = json.load(f)
            words = [word for word, count in vocab.items() for _ in range(count)]
        else:
            words = [word for tree in train_treebank for word in tree.words()]
        labels = [label for tree in train_treebank for label in tree.labels()]

        if self.parser_type == 'crf':
            words = [UNK, START, STOP] + words
            labels = [(DUMMY,)] + labels
        else:
            words = [UNK] + words

        word_vocab = Vocabulary.fromlist(words, unk_value=UNK)
        label_vocab = Vocabulary.fromlist(labels)

        if self.parser_type.endswith('rnng'):
            # Order is very important! See DiscParser/GenParser classes for why
            if self.parser_type == 'disc-rnng':
                actions = [SHIFT, REDUCE] + [NT(label) for label in label_vocab]
            elif self.parser_type == 'gen-rnng':
                actions = [REDUCE] + [NT(label) for label in label_vocab] + [GEN(word) for word in word_vocab]
            action_vocab = Vocabulary()
            for action in actions:
                action_vocab.add(action)
        else:
            action_vocab = Vocabulary()

        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.action_vocab = action_vocab

        self.train_treebank = train_treebank
        self.dev_treebank = dev_treebank
        self.test_treebank = test_treebank

        print('\n'.join((
            'Corpus statistics:',
            f'Vocab: {word_vocab.size:,} words, {label_vocab.size:,} nonterminals, {action_vocab.size:,} actions',
            f'Train: {len(train_treebank):,} sentences',
            f'Dev: {len(dev_treebank):,} sentences',
            f'Test: {len(test_treebank):,} sentences')))

    def build_model(self):
        assert self.word_vocab is not None, 'build corpus first'

        print('Initializing model...')
        self.model = dy.ParameterCollection()

        if self.parser_type == 'disc-rnng':
            parser = DiscRNNG(
                model=self.model,
                word_vocab=self.word_vocab,
                nt_vocab=self.label_vocab,
                action_vocab=self.action_vocab,
                word_emb_dim=self.word_emb_dim,
                nt_emb_dim=self.label_emb_dim,
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
        elif self.parser_type == 'gen-rnng':
            parser = GenRNNG(
                model=self.model,
                word_vocab=self.word_vocab,
                nt_vocab=self.label_vocab,
                action_vocab=self.action_vocab,
                word_emb_dim=self.word_emb_dim,
                nt_emb_dim=self.label_emb_dim,
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
        elif self.parser_type == 'crf':
            parser = ChartParser(
                model=self.model,
                word_vocab=self.word_vocab,
                label_vocab=self.label_vocab,
                word_embedding_dim=self.word_emb_dim,
                lstm_layers=self.lstm_layers,
                lstm_dim=self.lstm_dim,
                label_hidden_dim=self.label_hidden_dim,
                dropout=self.dropout,
            )
        self.parser = parser
        print('Number of parameters: {:,}'.format(self.parser.num_params))

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

        if self.parser_type == 'gen-rnng':
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
                print('| End of epoch {:3d}/{} | Elapsed {} | Current dev F1 {:4.2f} | Best dev F1 {:4.2f} (epoch {:2d})'.format(
                    epoch, self.max_epochs, self.timer.format_elapsed(), self.current_dev_fscore, self.best_dev_fscore, self.best_dev_epoch))
                print('-'*99)
        except KeyboardInterrupt:
            print('-'*99)
            print('Exiting from training early.')
            print('-'*99)

        self.check_dev()

        # Check test scores
        self.check_test()

        # Save model again but with test fscore
        self.save_checkpoint()

        # Save the losses for plotting and diagnostics
        self.write_losses()

        print('='*99)
        print('| End of training | Best dev F1 {:3.2f} (epoch {:2d}) | Test F1 {:3.2f}'.format(
            self.best_dev_fscore, self.best_dev_epoch, self.test_fscore))
        print('='*99)

    def train_epoch(self):
        """One epoch of sequential training."""
        self.parser.train()
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
            loss = dy.esum([self.parser.forward(tree) for tree in minibatch])
            loss /= self.batch_size

            # Add penalty if fine-tuning embeddings
            if self.fine_tune_embeddings:
                delta_penalty = self.parser.word_embedding.delta_penalty()
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

        dy.save(self.model_checkpoint_path, [self.parser])

        self.word_vocab.save(self.word_vocab_path)
        self.label_vocab.save(self.label_vocab_path)
        self.action_vocab.save(self.action_vocab_path)

        with open(self.state_checkpoint_path, 'w') as f:
            state = {
                'parser-type': self.parser_type,
                'num-params': int(self.parser.num_params),
                'num-epochs': self.current_epoch,
                'num-updates': self.num_updates,
                'current-lr': self.get_lr(),
                'best-dev-fscore': self.best_dev_fscore,
                'best-dev-epoch': self.best_dev_epoch,
                'test-fscore': self.test_fscore,
            }
            json.dump(state, f, indent=4)

    def load_checkpoint(self):
        self.model = dy.ParameterCollection()
        [self.parser] = dy.load(self.model_checkpoint_path, self.model)

    def predict(self, examples):
        self.parser.eval()
        trees = []
        for gold in tqdm(examples):
            dy.renew_cg()
            tree, *rest = self.parser.parse(gold.words())
            trees.append(tree.linearize())
        self.parser.train()
        return trees

    def check_dev(self):
        if self.parser_type in ('disc-rnng', 'crf'):
            self.check_dev_disc()
        if self.parser_type == 'gen-rnng':
            self.check_dev_gen()

    def check_test(self):
        if self.parser_type in ('disc-rnng', 'crf'):
            self.check_test_disc()
        if self.parser_type == 'gen-rnng':
            self.check_test_gen()

    def check_dev_disc(self):
        print('Evaluating F1 on development set...')

        # Predict trees
        trees = self.predict(self.dev_treebank)
        with open(self.dev_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)

        # Compute f-score
        dev_fscore = evalb(
            self.evalb_dir, self.dev_pred_path, self.dev_path, self.dev_result_path)

        # Log score to tensorboard
        self.tensorboard_writer.add_scalar(
            'dev/f-score', dev_fscore, self.num_updates)

        self.current_dev_fscore = dev_fscore
        if dev_fscore > self.best_dev_fscore:
            print(f'Saving new best model to `{self.model_checkpoint_path}`...')
            self.best_dev_epoch = self.current_epoch
            self.best_dev_fscore = dev_fscore
            self.save_checkpoint()

    def check_dev_gen(self):
        print('Evaluating F1 and perplexity on development set...')

        decoder = GenerativeDecoder(model=self.parser)

        trees, dev_perplexity = decoder.predict_from_proposal_samples(
            path=self.dev_proposal_samples)

        with open(self.dev_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)

        dev_fscore = evalb(
            self.evalb_dir, self.dev_pred_path, self.dev_path, self.dev_result_path)

        # Log score to tensorboard
        self.tensorboard_writer.add_scalar(
            'dev/f-score', dev_fscore, self.num_updates)
        self.tensorboard_writer.add_scalar(
            'dev/perplexity', dev_perplexity, self.num_updates)

        self.current_dev_fscore = dev_fscore
        self.dev_perplexity = dev_perplexity

        if dev_fscore > self.best_dev_fscore:
            print(f'Saving new best model to `{self.model_checkpoint_path}`...')
            self.best_dev_epoch = self.current_epoch
            self.best_dev_fscore = dev_fscore
            self.save_checkpoint()

    def check_test_disc(self):
        print('Evaluating F1 on test set...')

        print(f'Loading best saved model from `{self.model_checkpoint_path}` '
              f'(epoch {self.best_dev_epoch}, fscore {self.best_dev_fscore})...')
        self.load_checkpoint()

        # Predict trees.
        trees = self.predict(self.test_treebank)
        with open(self.test_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)

        # Compute f-score.
        test_fscore = evalb(
            self.evalb_dir, self.test_pred_path, self.test_path, self.test_result_path)

        self.tensorboard_writer.add_scalar(
            'test/f-score', test_fscore)

        self.test_fscore = test_fscore

    def check_test_gen(self):
        print('Evaluating F1 and perplexity on test set...')

        print(f'Loading best saved model from `{self.model_checkpoint_path}` '
              f'(epoch {self.best_dev_epoch}, fscore {self.best_dev_fscore})...')
        self.load_checkpoint()
        self.parser.eval()

        decoder = GenerativeDecoder(model=self.parser)

        trees, test_perplexity = decoder.predict_from_proposal_samples(
            path=self.test_proposal_samples)
        with open(self.test_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)

        # Compute f-score.
        test_fscore = evalb(
            self.evalb_dir, self.test_pred_path, self.test_path, self.test_result_path)

        # Log score to tensorboard.
        self.tensorboard_writer.add_scalar(
            'test/f-score', test_fscore)
        self.tensorboard_writer.add_scalar(
            'test/perplexity', test_perplexity)

        self.test_fscore = test_fscore
        self.test_perplexity = test_perplexity

    def write_losses(self):
        with open(self.loss_path, 'w') as f:
            print('loss', file=f)
            for loss in self.losses:
                print(loss, file=f)

    def finalize_model_folder(self):
        move_to_final_folder(
            self.subdir, self.model_path_base, self.best_dev_fscore)
