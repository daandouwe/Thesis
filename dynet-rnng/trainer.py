import os
import json
import itertools
from math import inf
from collections import Counter

import numpy as np
import dynet as dy
from tensorboardX import SummaryWriter

from vocabulary import Vocabulary, UNK
from actions import SHIFT, REDUCE, NT, GEN
from tree import fromstring
from decode import GenerativeDecoder
from model import DiscRNNG, GenRNNG
from eval import evalb
from utils import Timer, get_folders, write_args, ceil_div, replace_quotes, replace_brackets, unkify


class Trainer:
    """Trainer for RNNG."""
    def __init__(
            self,
            rnng_type='disc',
            args=None,
            train_path=None,
            dev_path=None,
            test_path=None,
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
        self.evalb_dir = evalb_dir
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
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
        self.word_vocab_path = os.path.join(checkdir, 'word-vocab.json')
        self.nt_vocab_path = os.path.join(checkdir, 'nt-vocab.json')
        self.action_vocab_path = os.path.join(checkdir, 'action-vocab.json')
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

        print(f'Loading test trees from `{self.dev_path}`...')
        with open(self.test_path) as f:
            test_treebank = [fromstring(line.strip()) for line in f]

        print("Constructing vocabularies...")
        words = [word for tree in train_treebank for word in tree.leaves()] + [UNK]

        ###########
        # words = [word for tree in train_treebank for word in tree.leaves()]
        # counts = Counter(words)
        # # if self.elaborate_unk:
        # if True:
        #     words = [word for word in words if counts[word] > 1] + \
        #             [unkify(word, counts) for word in words if counts[word] <= 1]
        # else:
        #     words = [word for word in words if counts[word] > self.threshold] + [UNK]
        ###########

        tags = [tag for tree in train_treebank for tag in tree.tags()]
        nonterminals = [label for tree in train_treebank for label in tree.labels()]

        word_vocab = Vocabulary.fromlist(words, unk=True)
        tag_vocab = Vocabulary.fromlist(tags)
        nt_vocab = Vocabulary.fromlist(nonterminals)

        # Order is very important, see DiscParser class
        if self.rnng_type == 'disc':
            actions = [SHIFT, REDUCE] + [NT(label) for label in nt_vocab]
        elif self.rnng_type == 'gen':
            actions = [REDUCE] + [NT(label) for label in nt_vocab] + [GEN(word) for word in word_vocab]
        action_vocab = Vocabulary()
        for action in actions:
            action_vocab.add(action)

        self.word_vocab = word_vocab
        self.nt_vocab = nt_vocab
        self.action_vocab = action_vocab

        self.train_treebank = train_treebank
        self.dev_treebank = dev_treebank
        self.test_treebank = test_treebank

        print('\n'.join((
            'Corpus statistics:',
            f'Vocab: {word_vocab.size:,} words ({len(word_vocab.unks)} UNK-types), {nt_vocab.size:,} nonterminals, {action_vocab.size:,} actions',
            f'Train: {len(train_treebank):,} sentences',
            f'Dev: {len(dev_treebank):,} sentences',
            f'Test: {len(test_treebank):,} sentences')))

    def build_model(self):
        assert self.word_vocab is not None, 'build corpus first'

        print('Initializing model...')
        self.model = dy.ParameterCollection()
        if self.rnng_type == 'disc':
            self.rnng = DiscRNNG(
                model=self.model,
                word_vocab=self.word_vocab,
                nt_vocab=self.nt_vocab,
                action_vocab=self.action_vocab,
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
                word_vocab=self.word_vocab,
                nt_vocab=self.nt_vocab,
                action_vocab=self.action_vocab,
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
            loss = dy.esum([self.rnng.forward(tree) for tree in minibatch])
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

                print('| step {:6d}/{:5d} ({:.0%}) | loss {:7.3f} | lr {:.1e} | {:4.1f} sents/sec | {:4.1f} updates/sec | elapsed {} | eta {} '.format(
                    step, num_batches, step/num_batches, avg_loss, lr, sents_per_sec, updates_per_sec,
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

        dy.save(self.model_checkpoint_path, [self.rnng])
        self.word_vocab.save(self.word_vocab_path)
        self.nt_vocab.save(self.nt_vocab_path)
        self.action_vocab.save(self.action_vocab_path)
        with open(self.state_checkpoint_path, 'w') as f:
            state = {
                'rnng-type': self.rnng_type,
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

    def predict(self, examples, proposal_samples=None):
        if self.rnng_type == 'disc':
            decoder = self.rnng
        elif self.rnng_type == 'gen':
            decoder = GenerativeDecoder(model=self.rnng)
            decoder.load_proposal_samples(path=proposal_samples)
        trees = []
        for i, gold in enumerate(examples):
            dy.renew_cg()
            tree, *rest = decoder.parse(gold.leaves())
            trees.append(tree.linearize())
            if i % 10 == 0:
                print(f'Predicting sentence {i}/{len(examples)}...', end='\r')
        return trees

    def check_dev(self):
        print('Evaluating F1 on development set...')
        self.rnng.eval()
        # Predict trees.
        dev_treebank = self.dev_treebank[:30] if self.rnng_type == 'gen' else self.dev_treebank # Is slooow!
        trees = self.predict(dev_treebank, proposal_samples=self.dev_proposal_samples)
        with open(self.dev_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)
        # Compute f-score.
        dev_fscore = evalb(
            self.evalb_dir, self.dev_pred_path, self.dev_path, self.dev_result_path)
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
        trees = self.predict(self.test_treebank, proposal_samples=self.test_proposal_samples)
        with open(self.test_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)
        # Compute f-score.
        test_fscore = evalb(
            self.evalb_dir, self.test_pred_path, self.test_path, self.test_result_path)
        self.tensorboard_writer.add_scalar('test/f-score', test_fscore)
        return test_fscore

    def write_losses(self):
        with open(self.loss_path, 'w') as f:
            print('loss', file=f)
            for loss in self.losses:
                print(loss, file=f)


class SemiSupervisedTrainer:

    def __init__(
            self,
            args=None,
            evalb_dir=None,
            unlabeled_path=None,
            train_path=None,
            dev_path=None,
            test_path=None,
            text_type=None,
            joint_model_path=None,
            post_model_path=None,
            lmbda=1.0,
            use_argmax_baseline=False,
            use_mean_baseline=False,
            use_lm_baseline=False,
            clip_learning_signal=-20,
            max_epochs=inf,
            max_time=inf,
            num_samples=3,
            alpha=None,
            lr=None,
            batch_size=1,
            weight_decay=None,
            lr_decay=None,
            lr_decay_patience=None,
            max_grad_norm=None,
            use_glove=False,
            glove_dir=None,
            print_every=1,
            eval_every=-1,  # default is every epoch (-1)
    ):
        self.args = args

        # Data
        self.evalb_dir = evalb_dir
        self.unlabeled_path = os.path.expanduser(unlabeled_path)
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.text_type = text_type

        # Model paths
        self.model = None  # will be a dynet ParameterCollection
        self.joint_model_path = joint_model_path
        self.post_model_path = post_model_path

        # Baselines
        self.lmbda = lmbda  # scaling coefficient for unsupervised objective
        self.use_argmax_baseline = use_argmax_baseline
        self.use_mean_baseline = use_mean_baseline
        self.use_lm_baseline = use_lm_baseline
        self.clip_learning_signal = clip_learning_signal

        # Training
        self.max_epochs = max_epochs
        self.max_time = max_time
        self.num_samples = num_samples
        self.alpha = alpha
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.lr_decay_patience = lr_decay_patience
        self.max_grad_norm = max_grad_norm
        self.use_glove = use_glove
        self.glove_dir = glove_dir
        self.print_every = print_every
        self.eval_every = eval_every

        self.num_updates = 0
        self.cum_learning_signal = 0
        self.losses = []
        self.sup_losses = []
        self.unsup_losses = []
        self.baseline_losses = []

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
        self.word_vocab_path = os.path.join(checkdir, 'word-vocab.json')
        self.nt_vocab_path = os.path.join(checkdir, 'nt-vocab.json')
        self.action_vocab_path = os.path.join(checkdir, 'action-vocab.json')
        self.loss_path = os.path.join(logdir, 'loss.csv')
        self.tensorboard_writer = SummaryWriter(logdir)

        # Dev paths
        self.dev_pred_path = os.path.join(outdir, 'dev.pred.trees')
        self.dev_result_path = os.path.join(outdir, 'dev.result')

        # Test paths
        self.test_pred_path = os.path.join(outdir, 'test.pred.trees')
        self.test_result_path = os.path.join(outdir, 'test.result')

    def build_corpus(self):
        # ###
        # print(f'Loading unsupervised data from `{self.unlabeled_path}`...')
        # with open(self.unlabeled_path) as f:
        #     unlabeled_data = [
        #         replace_brackets(replace_quotes(line.strip().split()))
        #         for line in f
        #     ]
        #
        # print(f'Loading training trees from `{self.train_path}`...')
        # with open(self.train_path) as f:
        #     train_treebank = [fromstring(line.strip()) for line in f]
        #
        # unsup_words = [word for line in unlabeled_data for word in line]
        # ptb_words = [word for tree in train_treebank for word in tree.leaves()]
        #
        # unsup_counts = Counter(unsup_words)
        # ptb_counts = Counter(ptb_words)
        #
        # unsup_words = set(unsup_words)
        # ptb_words = set(ptb_words)
        #
        # unsup_min_count = set([word for word in unsup_words if unsup_counts[word] > 1])
        # ptb_min_count = set([word for word in ptb_words if ptb_counts[word] > 1])
        #
        # print(f'unsup vocab: {len(unsup_words):,}')
        # print(f'ptb vocab: {len(ptb_words):,}')
        # print(f'ptb union unsup: {len(ptb_words.union(unsup_words)):,}')
        # print(f'ptb intersect unsup: {len(ptb_words.intersection(unsup_words)):,}')
        #
        # print(f'unsup > 1: {len(unsup_min_count):,}')
        # print(f'ptb > 1: {len(ptb_min_count):,}')
        # print(f'ptb union unsup > 1: {len(ptb_min_count.union(unsup_min_count)):,}')
        # print(f'ptb intersect unsup > 1: {len(ptb_min_count.intersection(unsup_min_count)):,}')
        # ###

        print(f'Loading training trees from `{self.train_path}`...')
        with open(self.train_path) as f:
            train_treebank = [fromstring(line.strip()) for line in f]

        print(f'Loading development trees from `{self.dev_path}`...')
        with open(self.dev_path) as f:
            dev_treebank = [fromstring(line.strip()) for line in f]

        print(f'Loading test trees from `{self.dev_path}`...')
        with open(self.test_path) as f:
            test_treebank = [fromstring(line.strip()) for line in f]

        print(f'Loading unlabeled data from `{self.unlabeled_path}`...')
        with open(self.unlabeled_path) as f:
            unlabeled_data = [replace_brackets(replace_quotes(line.strip().split())) for line in f]

        print("Constructing vocabularies...")
        words = [word for tree in train_treebank for word in tree.leaves()] + [UNK]
        tags = [tag for tree in train_treebank for tag in tree.tags()]
        nonterminals = [label for tree in train_treebank for label in tree.labels()]

        word_vocab = Vocabulary.fromlist(words, unk=True)
        tag_vocab = Vocabulary.fromlist(tags)
        nt_vocab = Vocabulary.fromlist(nonterminals)

        # Order is very important, see DiscParser class
        disc_actions = [SHIFT, REDUCE] + [NT(label) for label in nt_vocab]
        disc_action_vocab = Vocabulary()
        for action in disc_actions:
            disc_action_vocab.add(action)

        # Order is very important, see GenParser class
        gen_action_vocab = Vocabulary()
        gen_actions = [REDUCE] + [NT(label) for label in nt_vocab] + [GEN(word) for word in word_vocab]
        for action in gen_actions:
            gen_action_vocab.add(action)

        self.word_vocab = word_vocab
        self.nt_vocab = nt_vocab
        self.gen_action_vocab = gen_action_vocab
        self.disc_action_vocab = disc_action_vocab

        self.train_treebank = train_treebank
        self.dev_treebank = dev_treebank
        self.test_treebank = test_treebank
        self.unlabeled_data = unlabeled_data

        print('\n'.join((
            'Corpus statistics:',
            f'Vocab: {word_vocab.size:,} words ({len(word_vocab.unks)} UNK-types), {nt_vocab.size:,} nonterminals, {gen_action_vocab.size:,} actions',
            f'Train: {len(train_treebank):,} sentences',
            f'Dev: {len(dev_treebank):,} sentences',
            f'Test: {len(test_treebank):,} sentences',
            f'Unlabeled: {len(unlabeled_data):,} sentences')))

    def load_models(self):
        self.model = dy.ParameterCollection()
        self.load_joint_model()
        self.load_post_model()
        # for param in self.model.parameters_list():
            # print(param.name())

    def load_joint_model(self):
        assert self.model is not None, 'build model first'

        model_path = os.path.join(self.joint_model_path, 'model')
        state_path = os.path.join(self.joint_model_path, 'state.json')

        [self.joint_model] = dy.load(model_path, self.model)
        assert isinstance(self.joint_model, GenRNNG), type(self.joint_model)
        self.joint_model.train()

        with open(state_path) as f:
            state = json.load(f)
        epochs, fscore = state['epochs'], state['test-fscore']
        print(f'Loaded joint model trained for {epochs} epochs with test fscore {fscore}.')

    def load_post_model(self):
        assert self.model is not None, 'build model first'

        model_path = os.path.join(self.post_model_path, 'model')
        state_path = os.path.join(self.post_model_path, 'state.json')

        [self.post_model] = dy.load(model_path, self.model)
        assert isinstance(self.post_model, DiscRNNG), type(self.post_model)
        self.post_model.train()

        with open(state_path) as f:
            state = json.load(f)
        epochs, fscore = state['epochs'], state['test-fscore']
        print(f'Loaded post model trained for {epochs} epochs with test fscore {fscore}.')

    def build_optimizer(self):
        assert self.model is not None, 'build model first'

        if self.args.optimizer == 'sgd':
            self.optimizer = dy.SimpleSGDTrainer(self.model, learning_rate=self.lr)
            self.baseline_optimizer = dy.SimpleSGDTrainer(self.model, learning_rate=self.lr)
        elif self.args.optimizer == 'adam':
            self.optimizer = dy.AdamTrainer(self.model, alpha=self.lr)
            self.baseline_optimizer = dy.AdamTrainer(self.model, alpha=self.lr)

        self.optimizer.set_clip_threshold(self.max_grad_norm)
        self.baseline_optimizer.set_clip_threshold(self.max_grad_norm)
        self.model.set_weight_decay(self.weight_decay)

    def build_baseline_parameters(self):
        self.a_arg = self.post_model.model.add_parameters(1, init=1)
        self.c_arg = self.post_model.model.add_parameters(1, init=0)

    def batchify(self, data):
        batches = [data[i*self.batch_size:(i+1)*self.batch_size]
            for i in range(ceil_div(len(data), self.batch_size))]
        return batches

    def train(self):
        self.build_paths()
        self.build_corpus()
        self.load_models()
        self.build_baseline_parameters()
        self.build_optimizer()

        self.cum_learning_signal = 0
        self.num_updates = 0
        self.unlabeled_batches = iter(self.batchify(self.unlabeled_data))

        self.timer = Timer()
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
            # Check development f-score
            # self.check_dev_fscore()
            # Anneal learning rate depending on development set f-score
            # self.anneal_lr()
            print('-'*99)
            print('| End of epoch {:3d}/{} | Elapsed {} | Current dev F1 {:4.2f} | Best dev F1 {:4.2f} (epoch {:2d})'.format(
                epoch, self.max_epochs, self.timer.format_elapsed(), self.current_dev_fscore, self.best_dev_fscore, self.best_dev_epoch))
            print('-'*99)

    def train_epoch(self):

        def get_unlabeled_batch():
            try:
                batch = next(self.unlabeled_batches)
            except StopIteration:
                np.random.shuffle(self.unlabeled_data)
                self.unlabeled_batches = iter(self.batchify(self.unlabeled_data)))
                batch = next(self.unlabeled_batches)
            return batch

        # We loop over the labeled_batches and request an unlabeled batch
        labeled_batches = self.batchify(self.train_treebank)
        for i, labeled_batch in enumerate(labeled_batches):
            dy.renew_cg()

            sup_loss = self.supervised_step(labeled_batch)
            unsup_loss, baseline_loss = self.unsupervised_step(get_unlabeled_batch())

            loss = sup_loss + self.lmbda * unsup_loss

            # Optimize objective
            loss.forward()
            loss.backward()
            self.optimizer.update()

            # Optimize baseline
            baseline_loss.forward()
            baseline_loss.backward()
            self.baseline_optimizer.update()

            # Store losses
            self.losses.append(loss.value())
            self.sup_losses.append(sup_loss.value())
            self.unsup_losses.append(unsup_loss.value())
            self.baseline_losses.append(baseline_loss.value())

            self.num_updates += 1

            if self.num_updates % self.print_every == 0:
                loss = np.mean(self.losses[-self.print_every:])
                sup_loss = np.mean(self.sup_losses[-self.print_every:])
                unsup_loss = np.mean(self.unsup_losses[-self.print_every:])
                baseline_loss = np.mean(self.baseline_losses[-self.print_every:])

                self.tensorboard_writer.add_scalar(
                    'semisup/loss/total', loss, self.num_updates)
                self.tensorboard_writer.add_scalar(
                    'semisup/loss/sup', sup_loss, self.num_updates)
                self.tensorboard_writer.add_scalar(
                    'semisup/loss/unsup', unsup_loss, self.num_updates)
                self.tensorboard_writer.add_scalar(
                    'semisup/loss/baseline', baseline_loss, self.num_updates)

                self.tensorboard_writer.add_scalar(
                    'semisup/baseline/argmax-a', self.a_arg.value(), self.num_updates)
                self.tensorboard_writer.add_scalar(
                    'semisup/baseline/argmax-c', self.c_arg.value(), self.num_updates)

                print('| epoch {:2d} | step {:4d} | loss {:6.3f} | sup-loss {:5.3f} | unsup-loss {:6.3f} | baseline-loss {:3.3f} '.format(
                    self.current_epoch, self.num_updates, loss, sup_loss, unsup_loss, baseline_loss))

            if self.num_updates % self.eval_every == 0 and self.eval_every != -1:
                fscore = self.check_dev_fscore()
                pp = self.check_dev_perplexity()
                print(89*'=')
                print('| dev F1 {:4.2f} | dev perplexity {:4.2f}'.format(
                    fscore, pp))
                print(89*'=')

    def supervised_step(self, batch):
        losses = []
        for tree in batch:
            # loss = self.joint_model.forward(tree) + self.post_model.forward(tree)
            loss = self.joint_model.forward(tree)
            losses.append(loss)
        loss = dy.esum(losses) / self.batch_size
        return loss

    def unsupervised_step(self, batch):
        losses = []
        baseline_losses = []
        batch_learning_signal = 0
        histogram = dict(signal=[], centered=[], post=[], joint=[]) # for tensorboard histogram
        for words in batch:
            # Get samples
            trees, _ = self.sample(words)

            # Compute mean \mean_y [log p(x,y)] and \mean_y [log q(y|x)]
            joint_logprob = self.joint(trees)
            post_logprob = self.posterior(trees)

            # Compute \mean_y [log p(x,y) - log q(y|x)] and detach
            learning_signal = blockgrad(joint_logprob - post_logprob)

            # Compute baseline
            baselines = self.baselines(words)
            centered_learning_signal = learning_signal - baselines

            # Optional clipping of learning signal
            if self.clip_learning_signal is not None:
                if centered_learning_signal.value() < self.clip_learning_signal:
                    centered_learning_signal = dy.scalarInput(self.clip_learning_signal)

            post_loss = -blockgrad(centered_learning_signal) * post_logprob
            joint_loss = -joint_logprob
            loss = post_loss + joint_loss
            losses.append(loss)

            baseline_loss = centered_learning_signal**2
            baseline_losses.append(baseline_loss)

            histogram['signal'].append(learning_signal)
            histogram['centered'].append(centered_learning_signal.value())
            histogram['post'].append(post_logprob.value())
            histogram['joint'].append(joint_logprob.value())

            batch_learning_signal += centered_learning_signal.value()

        self.cum_learning_signal += batch_learning_signal / self.batch_size

        self.tensorboard_writer.add_histogram(
            'semisup/unsup/learning-signal', np.array(histogram['signal']), self.num_updates)
        self.tensorboard_writer.add_histogram(
            'semisup/unsup/centered-learning-signal', np.array(histogram['centered']), self.num_updates)
        self.tensorboard_writer.add_histogram(
            'semisup/unsup/post-logprob', np.array(histogram['post']), self.num_updates)
        self.tensorboard_writer.add_histogram(
            'semisup/unsup/joint-logprob', np.array(histogram['joint']), self.num_updates)

        loss = dy.esum(losses) / self.batch_size
        baseline_loss = dy.esum(baseline_losses) / self.batch_size

        return loss, baseline_loss

    def sample(self, words):
        samples = [self.post_model.sample(words, self.alpha) for _ in range(self.num_samples)]
        trees, nlls = zip(*samples)
        return trees, nlls

    def joint(self, trees):
        logprobs = [-self.joint_model.forward(tree) for tree in trees]
        return dy.esum(logprobs) / len(logprobs)

    def posterior(self, trees):
        logprobs = [-self.post_model.forward(tree) for tree in trees]
        return dy.esum(logprobs) / len(logprobs)

    def baselines(self, words):
        b = 0
        if self.use_mean_baseline:
            b += self.mean_baseline()
        if self.use_lm_baseline:
            b += self.lm_baseline(words)
        if self.use_argmax_baseline:
            b += self.argmax_baseline(words)
        return b

    def mean_baseline(self):
        if self.num_updates == 0:
            return 0
        else:
            baseline = self.cum_learning_signal / self.num_updates

            # self.tensorboard_writer.add_scalar(
            #     'semisup/baseline/mean-baseline', baseline, self.num_updates)
            return mean_baseline

    def argmax_baseline(self, words):
        tree, _ = self.post_model.parse(words)
        joint_logprob = -self.joint_model.forward(tree)
        baseline = self.a_arg * blockgrad(joint_logprob) + self.c_arg

        # self.tensorboard_writer.add_scalar(
        #     'semisup/baseline/argmax-baseline', baseline.value(), self.num_updates)
        # self.tensorboard_writer.add_scalar(
        #     'semisup/baseline/argmax-logprob', joint_logprob.value(), self.num_updates)
        # self.tensorboard_writer.add_scalar(
        #     'semisup/baseline/a_arg', self.a_arg.value(), self.num_updates)
        # self.tensorboard_writer.add_scalar(
        #     'semisup/baseline/c_arg', self.c_arg.value(), self.num_updates)

        return baseline

    def predict(self, examples):
        self.post_model.eval()
        trees = []
        for i, tree in enumerate(examples):
            dy.renew_cg()
            tree, *rest = self.post_model.parse(list(tree.leaves()))
            trees.append(tree.linearize())
            if i % 10 == 0:
                print(f'Predicting sentence {i}/{len(examples)}...', end='\r')
        self.post_model.train()
        return trees

    def check_dev_fscore(self):
        print('Evaluating F1 on development set...')
        # Predict trees.
        trees = self.predict(self.dev_treebank)
        with open(self.dev_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)
        # Compute f-score.
        dev_fscore = evalb(
            self.evalb_dir, self.dev_pred_path, self.dev_path, self.dev_result_path)
        # Log score to tensorboard.
        self.current_dev_fscore = dev_fscore
        self.tensorboard_writer.add_scalar('semisup/dev/f-score', dev_fscore, self.num_updates)
        return dev_fscore

    def check_dev_perplexity(self):
        print('Evaluating perplexity on development set...')

        decoder = GenerativeDecoder(
            model=self.joint_model,
            proposal=self.post_model,
            num_samples=10,
            alpha=1.
        )
        examples = self.dev_treebank[:100]
        pp = 0.
        for i, tree in enumerate(examples):
            pp += decoder.perplexity(list(tree.leaves()))
            if i % 10 == 0:
                print(f'Predicting sentence {i}/{len(examples)}...', end='\r')
        avg_pp = pp / len(examples)
        self.tensorboard_writer.add_scalar('semisup/dev/perplexity', avg_pp, self.num_updates)
        return avg_pp


def blockgrad(expression):
    """Detach the expression from the computation graph"""
    return expression.value()
