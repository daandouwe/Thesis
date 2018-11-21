import os
import json
import itertools
from math import inf
from collections import Counter

import numpy as np
import dynet as dy
from tensorboardX import SummaryWriter
from tqdm import tqdm

from vocabulary import Vocabulary, UNK
from actions import SHIFT, REDUCE, NT, GEN
from tree import fromstring
from decode import GenerativeDecoder
from model import DiscRNNG, GenRNNG
from feedforward import Feedforward, Affine
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
        subdir, logdir, checkdir, outdir = get_folders(self.args)
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
            if self.rnng_type == 'gen':
                self.check_dev_perplexity()

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
        for gold in tqdm(examples):
            dy.renew_cg()
            tree, *rest = decoder.parse(gold.leaves())
            trees.append(tree.linearize())
        return trees

    def check_dev(self):
        print('Evaluating F1 on development set...')
        self.rnng.eval()

        # Predict trees.
        trees = self.predict(self.dev_treebank, proposal_samples=self.dev_proposal_samples)
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

    def check_dev_perplexity(self):
        print('Evaluating perplexity on development set...')

        decoder = GenerativeDecoder(model=self.rnng)
        decoder.load_proposal_samples(path=self.dev_proposal_samples)

        pp = 0.
        for tree in tqdm(self.dev_treebank):
            dy.renew_cg()
            pp += decoder.perplexity(list(tree.leaves()))
        avg_pp = pp / len(self.dev_treebank)

        self.tensorboard_writer.add_scalar('dev/perplexity', avg_pp)

        return avg_pp

    def check_test_perplexity(self):
        print('Evaluating perplexity on test set...')

        decoder = GenerativeDecoder(model=self.rnng)
        decoder.load_proposal_samples(path=self.test_proposal_samples)

        pp = 0.
        for tree in tqdm(self.test_treebank):
            dy.renew_cg()
            pp += decoder.perplexity(list(tree.leaves()))
        avg_pp = pp / len(self.test_treebank)

        self.tensorboard_writer.add_scalar(
            'test/perplexity', avg_pp, self.num_updates)

        return avg_pp

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
            joint_model_path=None,
            post_model_path=None,
            lmbda=1.0,
            use_argmax_baseline=False,
            use_mlp_baseline=False,
            clip_learning_signal=-20,
            max_unlabeled_sent_len=40,
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
            eval_at_start=False
    ):
        self.args = args

        # Data
        self.evalb_dir = evalb_dir
        self.unlabeled_path = os.path.expanduser(unlabeled_path)
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path

        # Model paths
        self.model = None  # will be a dynet ParameterCollection
        self.joint_model_path = joint_model_path
        self.post_model_path = post_model_path

        # Baselines
        self.lmbda = lmbda  # scaling coefficient for unsupervised objective
        self.use_argmax_baseline = use_argmax_baseline
        self.use_mlp_baseline = use_mlp_baseline
        self.clip_learning_signal = clip_learning_signal
        self.max_unlabeled_sent_len = max_unlabeled_sent_len

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
        self.eval_at_start = eval_at_start

        self.num_updates = 0
        self.learning_signals = []
        self.baseline_values = []
        self.centered_learning_signals = []
        self.losses = []
        self.sup_losses = []
        self.unsup_losses = []
        self.baseline_losses = []

    def build_paths(self):
        # Make output folder structure
        subdir, logdir, checkdir, outdir = get_folders(self.args)
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
        self.post_model_checkpoint_path = os.path.join(checkdir, 'post-model')
        self.joint_model_checkpoint_path = os.path.join(checkdir, 'joint-model')
        self.post_state_checkpoint_path = os.path.join(checkdir, 'post-state.json')
        self.joint_state_checkpoint_path = os.path.join(checkdir, 'joint-state.json')
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
            unlabeled_data = [
                replace_brackets(replace_quotes(line.strip().split()))
                for line in f
                if len(line.split()) < self.max_unlabeled_sent_len
            ]

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
            self.baseline_optimizer = dy.SimpleSGDTrainer(self.baseline_model, learning_rate=10*self.lr)

        elif self.args.optimizer == 'adam':
            self.optimizer = dy.AdamTrainer(self.model, alpha=self.lr)
            self.baseline_optimizer = dy.AdamTrainer(self.baseline_model, alpha=10*self.lr)

        self.optimizer.set_clip_threshold(self.max_grad_norm)
        self.baseline_optimizer.set_clip_threshold(self.max_grad_norm)
        self.model.set_weight_decay(self.weight_decay)

    def build_baseline_parameters(self):
        self.baseline_model = dy.ParameterCollection()

        emb_dim = self.post_model.buffer_encoder.hidden_size
        self.gating = Feedforward(
            self.baseline_model, emb_dim, [128], 1)
        self.feedforward = Feedforward(
            self.baseline_model, emb_dim, [128], 1)

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

        self.num_updates = 0
        self.unlabeled_batches = iter(self.batchify(self.unlabeled_data))

        self.timer = Timer()

        if self.eval_at_start:
            print('Evaluating at start...')

            fscore = self.check_dev_fscore()
            pp = self.check_dev_perplexity()

            print(89*'=')
            print('| Start | dev F1 {:4.2f} | dev perplexity {:4.2f}'.format(
                fscore, pp))
            print(89*'=')


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

            print('='*89)
            print(f'| End of epoch {epoch} | elapsed {self.timer.elapsed()}')
            print('='*89)

    def train_epoch(self):

        def get_unlabeled_batch():
            """Infinite generator over unlabeled batches."""
            try:
                batch = next(self.unlabeled_batches)
            except StopIteration:
                np.random.shuffle(self.unlabeled_data)
                self.unlabeled_batches = iter(self.batchify(self.unlabeled_data))
                batch = next(self.unlabeled_batches)
            return batch

        labeled_batches = self.batchify(self.train_treebank)

        # We loop over the labeled_batches and request an unlabeled batch
        for i, labeled_batch in enumerate(labeled_batches, 1):

            if self.timer.elapsed() > self.max_time:
                break

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

            if i % self.print_every == 0:
                loss = np.mean(self.losses[-self.print_every:])
                sup_loss = np.mean(self.sup_losses[-self.print_every:])
                unsup_loss = np.mean(self.unsup_losses[-self.print_every:])
                baseline_loss = np.mean(self.baseline_losses[-self.print_every:])

                self.write_tensoboard_losses(loss, sup_loss, unsup_loss, baseline_loss)

                print('| epoch {:3d} | step {:5d}/{:5d} ({:.1%}) | loss {:6.3f} | sup-loss {:5.3f} | unsup-loss {:6.3f} | baseline-loss {:3.3f} | elapsed {} | {:.2f} updates/sec | eta {}'.format(
                    self.current_epoch, i, len(labeled_batches), i / len(labeled_batches),
                    loss, sup_loss, unsup_loss, baseline_loss,
                    self.timer.format_elapsed_epoch(), i / self.timer.elapsed_epoch(),
                    self.timer.format_eta(i, len(labeled_batches))))

            if self.num_updates % self.eval_every == 0 and self.eval_every != -1:
                fscore = self.check_dev_fscore()
                pp = self.check_dev_perplexity()
                print(89*'=')
                print('| dev F1 {:4.2f} | dev perplexity {:4.2f}'.format(
                    fscore, pp))
                print(89*'=')
                self.timer.new_epoch()
                self.save_checkpoint(fscore, pp)

    def supervised_step(self, batch):
        losses = []
        for tree in batch:
            loss = self.joint_model.forward(tree) + 0.1 * self.post_model.forward(tree)
            # loss = self.joint_model.forward(tree)
            losses.append(loss)
        loss = dy.esum(losses) / self.batch_size
        return loss

    def unsupervised_step(self, batch):
        losses = []
        baseline_losses = []
        # For various types of logging
        histogram = dict(
            signal=[],
            baselines=[],
            centered=[],
            normalized=[],
            post=[],
            joint=[],
            unique=[],
            scale=[]
        )

        for words in batch:
            # Get samples with their \mean_y [log q(y|x)] for samples y
            trees, post_logprob = self.sample(words)

            # Compute mean \mean_y [log p(x,y)] for samples y
            joint_logprob = self.joint(trees)

            # Compute \mean_y [log p(x,y) - log q(y|x)] and detach
            learning_signal = blockgrad(joint_logprob - post_logprob)

            # Substract baseline
            baselines = self.baselines(words)
            a = self.optimal_baseline_scale()
            centered_learning_signal = learning_signal - a * baselines

            # Normalize
            normalized_learning_signal = self.normalize(centered_learning_signal)

            # Optional clipping of learning signal
            if self.clip_learning_signal is not None:
                if normalized_learning_signal.value() < self.clip_learning_signal:
                    normalized_learning_signal = dy.scalarInput(self.clip_learning_signal)

            baseline_loss = centered_learning_signal**2
            post_loss = -blockgrad(normalized_learning_signal) * post_logprob
            joint_loss = -joint_logprob
            loss = post_loss + joint_loss

            losses.append(loss)
            baseline_losses.append(baseline_loss)

            # For tesorboard logging
            histogram['signal'].append(learning_signal)
            histogram['baselines'].append(baselines.value())
            histogram['scale'].append(a)
            histogram['centered'].append(centered_learning_signal.value())
            histogram['normalized'].append(normalized_learning_signal.value())
            histogram['post'].append(post_logprob.value())
            histogram['joint'].append(joint_logprob.value())
            histogram['unique'].append(len(set(tree.linearize() for tree in trees)))

        self.write_tensoboard_histogram(histogram)

        self.learning_signals.append(np.mean(histogram['signal']))
        self.baseline_values.append(np.mean(histogram['baselines']))
        self.centered_learning_signals.append(np.mean(histogram['centered']))

        # Average losses over minibatch
        loss = dy.esum(losses) / self.batch_size
        baseline_loss = dy.esum(baseline_losses) / self.batch_size

        return loss, baseline_loss

    def sample(self, words):
        samples = [self.post_model.sample(words, self.alpha) for _ in range(self.num_samples)]
        trees, nlls = zip(*samples)
        logprob = dy.esum([-nll for nll in nlls]) / len(nlls)
        return trees, logprob

    def joint(self, trees):
        logprobs = [-self.joint_model.forward(tree) for tree in trees]
        return dy.esum(logprobs) / len(logprobs)

    def posterior(self, trees):
        logprobs = [-self.post_model.forward(tree) for tree in trees]
        return dy.esum(logprobs) / len(logprobs)

    def baselines(self, words):
        b = dy.scalarInput(0)
        if self.use_mlp_baseline:
            b += self.mlp_baseline(words)
        if self.use_argmax_baseline:
            b += self.argmax_baseline(words)
        return b

    def argmax_baseline(self, words):
        """Parameter-free baseline based on argmax decoding."""
        tree, post_nll = self.post_model.parse(words)
        post_logprob = -post_nll
        joint_logprob = -self.joint_model.forward(tree)
        return dy.scalarInput(blockgrad(joint_logprob - post_logprob))

    def mlp_baseline(self, words):
        """Baseline parametrized by a feedfoward network."""
        # Get the word embeddings from the posterior model
        embeddings = [
            self.post_model.word_embedding(word_id)
            for word_id in self.post_model.word_vocab.indices(words)]
        # Obtain an rnn from the rnn builder of the posterior model's buffer encoder
        rnn = self.post_model.buffer_encoder.rnn_builder.initial_state()
        # Use this rnn to compute rnn embeddings (In reverse! See class `Buffer`.)
        encodings = rnn.transduce(reversed(embeddings))

        # Detach the embeddings from the computation graph
        encodings = [dy.inputTensor(blockgrad(encoding)) for encoding in encodings]

        # Compute gated sum to give one sentence encoding
        gates = [dy.logistic(self.gating(encoding)) for encoding in encodings]
        encoding = dy.esum([
            dy.cmult(gate, encoding)
            for gate, encoding in zip(gates, encodings)])

        # Compute scalar baseline-value using sentence encoding
        baseline = self.feedforward(encoding)

        # For fun
        # print('>', ' '.join('{} ({})'.format(
            # word, round(gate.value(), 1)) for word, gate in zip(words, reversed(gates))))

        return baseline

    def optimal_baseline_scale(self, n=500):
        """Estimate optimal scalar value for baseline."""
        if self.use_mlp_baseline:
            # baseline is trainable so scaling messes it up
            return 1.
        elif self.num_updates < 4:
            # cannot estimate variance
            return 1.
        elif self.use_argmax_baseline:
            # static baseline: optimal scaling exists
            baseline_values = np.array(self.baseline_values[-n:])
            signal_values = np.array(self.learning_signals[-n:])
            _, cov, _, var = np.cov([signal_values, baseline_values]).ravel()
            return cov/var
        else:
            # no baseline no scaling
            return 1.

    def normalize(self, signal):
        """Normalize the centered learning-signal."""
        signal_mean = np.mean(self.centered_learning_signals) if self.num_updates > 0 else 0.
        signal_var = np.var(self.centered_learning_signals) if self.num_updates > 1 else 1.
        return (signal - signal_mean) / np.sqrt(signal_var)

    def baseline_signal_covariance(self, n=200):
        """Estimate covariance between baseline and learning signal."""
        baseline_values = np.array(self.baseline_values[-n:])
        signal_values = np.array(self.learning_signals[-n:])

        baseline_mean = np.mean(baseline_values) if self.num_updates > 1 else 0.
        baseline_var = np.var(baseline_values) if self.num_updates > 2 else 1.

        signal_mean = np.mean(signal_values) if self.num_updates > 1 else 0.
        signal_var = np.var(signal_values) if self.num_updates > 2 else 1.

        if self.num_updates > 2:
            cov = np.mean((baseline_values - baseline_mean) * (signal_values - signal_mean))
        else:
            cov = 0
        corr = cov / (np.sqrt(signal_var) * np.sqrt(baseline_var))

        return cov, corr

    def write_tensoboard_losses(self, loss, sup_loss, unsup_loss, baseline_loss):
        self.tensorboard_writer.add_scalar(
            'semisup/loss/total', loss, self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/loss/sup', sup_loss, self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/loss/unsup', unsup_loss, self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/loss/baseline', baseline_loss, self.num_updates)

    def write_tensoboard_histogram(self, histogram):
        # Write batch values as histograms
        self.tensorboard_writer.add_histogram(
            'semisup/histogram/learning-signal', np.array(histogram['signal']), self.num_updates)
        self.tensorboard_writer.add_histogram(
            'semisup/histogram/baselines', np.array(histogram['baselines']), self.num_updates)
        self.tensorboard_writer.add_histogram(
            'semisup/histogram/scale', np.array(histogram['scale']), self.num_updates)
        self.tensorboard_writer.add_histogram(
            'semisup/histogram/centered-learning-signal', np.array(histogram['centered']), self.num_updates)
        self.tensorboard_writer.add_histogram(
            'semisup/histogram/normalized-learning-signal', np.array(histogram['normalized']), self.num_updates)
        self.tensorboard_writer.add_histogram(
            'semisup/histogram/post-logprob', np.array(histogram['post']), self.num_updates)
        self.tensorboard_writer.add_histogram(
            'semisup/histogram/joint-logprob', np.array(histogram['joint']), self.num_updates)
        self.tensorboard_writer.add_histogram(
            'semisup/histogram/unique-samples', np.array(histogram['unique']), self.num_updates)

        # Write batch means as scalar
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/learning-signal', np.mean(histogram['signal']), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/baselines', np.mean(histogram['baselines']), self.num_updates)
        self.tensorboard_writer.add_histogram(
            'semisup/unsup/scale', np.mean(histogram['scale']), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/centered-learning-signal', np.mean(histogram['centered']), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/normalized-learning-signal', np.mean(histogram['normalized']), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/post-logprob', np.mean(histogram['post']), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/joint-logprob', np.mean(histogram['joint']), self.num_updates)

        # Write signal statistics
        cov, corr = self.baseline_signal_covariance()
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/signal-mean', np.mean(self.learning_signals), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/signal-variance', np.var(self.learning_signals), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/centered-signal-mean', np.mean(self.centered_learning_signals), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/centered-signal-variance', np.var(self.centered_learning_signals), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/baseline-mean', np.mean(self.baseline_values), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/baseline-variance', np.var(self.baseline_values), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/signal-baseline-cov', cov, self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/signal-baseline-cor', corr, self.num_updates)
        self.tensorboard_writer.add_scalar(  # var(f') / var(f) = 1 - corr(f, b)**2
            'semisup/unsup/signal-baseline-var-frac', 1 - corr**2, self.num_updates)

    def predict(self, examples):
        self.post_model.eval()
        trees = []
        for tree in tqdm(examples):
            dy.renew_cg()
            tree, *rest = self.post_model.parse(list(tree.leaves()))
            trees.append(tree.linearize())
        self.post_model.train()
        return trees

    def check_dev_fscore(self):
        print('Evaluating F1 on development set...')
        # Predict trees.
        trees = self.predict(self.dev_treebank)
        with open(self.dev_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)

        print('Computing fscore...')

        # Compute f-score.
        dev_fscore = evalb(
            self.evalb_dir, self.dev_pred_path, self.dev_path, self.dev_result_path)

        print('Computed fscore.')

        # Log score to tensorboard.
        self.current_dev_fscore = dev_fscore
        self.tensorboard_writer.add_scalar(
            'semisup/dev/f-score', dev_fscore, self.num_updates)

        return dev_fscore

    def check_dev_perplexity(self):
        print('Evaluating perplexity on development set...')

        decoder = GenerativeDecoder(
            model=self.joint_model,
            proposal=self.post_model,
            num_samples=100,
            alpha=1.
        )

        pp = 0.
        for tree in tqdm(self.dev_treebank):
            dy.renew_cg()
            pp += decoder.perplexity(list(tree.leaves()))
        avg_pp = pp / len(self.dev_treebank)

        self.tensorboard_writer.add_scalar(
            'semisup/dev/perplexity', avg_pp, self.num_updates)

        return avg_pp

    def save_checkpoint(self, fscore, perplexity):
        dy.save(self.post_model_checkpoint_path, [self.post_model])
        dy.save(self.joint_model_checkpoint_path, [self.joint_model])

        self.word_vocab.save(self.word_vocab_path)
        self.nt_vocab.save(self.nt_vocab_path)
        self.action_vocab.save(self.action_vocab_path)

        with open(self.joint_state_checkpoint_path, 'w') as f:
            state = {
                'rnng-type': 'gen',
                'epochs': self.current_epoch,
                'num-updates': self.num_updates,
                'dev-perplexity': perplexity,
            }
            json.dump(state, f, indent=4)

        with open(self.post_state_checkpoint_path, 'w') as f:
            state = {
                'rnng-type': 'disc',
                'epochs': self.current_epoch,
                'num-updates': self.num_updates,
                'dev-fscore': fscore,
            }
            json.dump(state, f, indent=4)


class WakeSleepTrainer:

    def __init__(
            self,
            args=None,
            evalb_dir=None,
            unlabeled_path=None,
            train_path=None,
            dev_path=None,
            test_path=None,
            joint_model_path=None,
            post_model_path=None,
            max_unlabeled_sent_len=40,
            gamma_wake=0.5,
            gamma_sleep=0.5,
            max_epochs=inf,
            max_time=inf,
            max_lines=-1,
            lr=None,
            batch_size=1,
            weight_decay=None,
            lr_decay=None,
            lr_decay_patience=None,
            max_grad_norm=None,
            use_glove=False,
            glove_dir=None,
            print_every=1,
            eval_every=-1,
            eval_at_start=False
    ):
        self.args = args

        # Data
        self.evalb_dir = evalb_dir
        self.unlabeled_path = os.path.expanduser(unlabeled_path)
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.max_unlabeled_sent_len = max_unlabeled_sent_len

        # Model paths
        self.model = None  # will be a dynet ParameterCollection
        self.joint_model_path = joint_model_path
        self.post_model_path = post_model_path

        # Wake sleep parameters
        self.gamma_wake = gamma_wake
        self.gamma_sleep = gamma_sleep
        self.num_samples = int(ceil_div(batch_size, 1./gamma_wake))

        # Training
        self.max_epochs = max_epochs
        self.max_time = max_time
        self.max_lines = max_lines
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
        self.eval_at_start = eval_at_start

        self.num_sleep_updates = 0
        self.num_wake_updates = 0
        self.sleep_losses = dict(total=[], labeled=[], sampled=[])
        self.wake_losses = dict(total=[], labeled=[], sampled=[])

    def build_paths(self):
        # Make output folder structure
        subdir, logdir, checkdir, outdir = get_folders(self.args)
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

        print(f'Loading unlabeled data from `{self.unlabeled_path}`...')
        with open(self.unlabeled_path) as f:
            unlabeled_data = [
                replace_brackets(replace_quotes(line.strip().split()))
                for line in f
                if len(line.split()) < self.max_unlabeled_sent_len
            ]

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
        elif self.args.optimizer == 'adam':
            self.optimizer = dy.AdamTrainer(self.model, alpha=self.lr)

        self.optimizer.set_clip_threshold(self.max_grad_norm)
        self.model.set_weight_decay(self.weight_decay)

    def batchify(self, data):
        batches = [data[i*self.batch_size:(i+1)*self.batch_size]
            for i in range(ceil_div(len(data), self.batch_size))]
        return batches

    def train(self):
        self.build_paths()
        self.build_corpus()
        self.load_models()
        self.build_optimizer()

        self.unlabeled_data_iterator = iter(self.unlabeled_data)

        self.timer = Timer()

        if self.eval_at_start:
            print('Evaluating at start...')

            fscore = self.check_dev_fscore()
            pp = self.check_dev_perplexity()

            print(89*'=')
            print('| Start | dev F1 {:4.2f} | dev perplexity {:4.2f}'.format(
                fscore, pp))
            print(89*'=')

        print('Start training...')
        for epoch in itertools.count(start=1):

            if epoch > self.max_epochs:
                break
            if self.timer.elapsed() > self.max_time:
                break

            self.current_epoch = epoch

            # Wake phase over the dataset
            np.random.shuffle(self.train_treebank)
            self.sleep_epoch()

            # Sleep phase over the dataset
            np.random.shuffle(self.train_treebank)
            self.wake_epoch()

            # Check progress
            fscore = self.check_dev_fscore()
            pp = self.check_dev_perplexity()

            print(89*'=')
            print('| End of epoch {:3d}/{} | dev F1 {:4.2f} | dev perplexity {:4.2f} | elapsed {}'.format(
                 epoch, self.max_epochs, fscore, pp, self.timer.format_elapsed()))
            print(89*'=')

    def sleep_epoch(self):

        if self.max_lines != -1:
            labeled_batches = self.batchify(
                np.random.choice(self.train_treebank, size=self.max_lines))
        else:
            labeled_batches = self.batchify(self.train_treebank)

        sleep_timer = Timer()

        # We loop over the labeled_batches and request unlabeled data as we go
        for i, labeled_batch in enumerate(labeled_batches, 1):

            dy.renew_cg()

            # Sample (x_i, y_i) ~ p(x,y) iid
            sampled_batch = [
                self.joint_model.sample()[0]  # select only the tree
                for _ in range(self.num_samples)]

            labeled_loss = dy.esum([self.post_model.forward(tree) for tree in labeled_batch])
            sampled_loss = dy.esum([self.post_model.forward(tree) for tree in sampled_batch])

            loss = (labeled_loss + sampled_loss) / (self.batch_size + self.num_samples)

            # Optimize objective
            loss.forward()
            loss.backward()
            self.optimizer.update()

            # Store losses
            self.sleep_losses['total'].append(loss.value())
            self.sleep_losses['labeled'].append(labeled_loss.value() / self.batch_size)
            self.sleep_losses['sampled'].append(sampled_loss.value() / self.num_samples)

            self.num_sleep_updates += 1

            if self.num_sleep_updates % self.print_every == 0:
                loss = np.mean(self.sleep_losses['total'][-self.print_every:])
                labeled_loss = np.mean(self.sleep_losses['labeled'][-self.print_every:])
                sampled_loss = np.mean(self.sleep_losses['sampled'][-self.print_every:])

                self.tensorboard_writer.add_scalar(
                    'wake-sleep/sleep/loss', loss, self.num_sleep_updates)
                self.tensorboard_writer.add_scalar(
                    'wake-sleep/sleep/loss-labeled', labeled_loss, self.num_sleep_updates)
                self.tensorboard_writer.add_scalar(
                    'wake-sleep/sleep/loss-sampled', sampled_loss, self.num_sleep_updates)

                print('| epoch {:2d} | sleep phase | update {:4d}/{:5d} ({:.1%}) | loss {:6.3f} | elapsed {} | {:.2f} updates/sec | eta {}'.format(
                    self.current_epoch, i, len(labeled_batches), i / len(labeled_batches), loss,
                    self.timer.format_elapsed(), i / sleep_timer.elapsed(),
                    self.timer.format((len(labeled_batches) - i) / i * sleep_timer.elapsed())))

    def wake_epoch(self):

        def get_unlabeled_sentence():
            try:
                sentence = next(self.unlabeled_data_iterator)
            except StopIteration:
                np.random.shuffle(self.unlabeled_data)
                self.unlabeled_data_iterator = iter(self.unlabeled_data)
                sentence = next(self.unlabeled_data_iterator)
            return sentence

        if self.max_lines != -1:
            labeled_batches = self.batchify(
                np.random.choice(self.train_treebank, size=self.max_lines))
        else:
            labeled_batches = self.batchify(self.train_treebank)

        wake_timer = Timer()

        # We loop over the labeled_batches and request unlabeled data as we go
        for i, labeled_batch in enumerate(labeled_batches, 1):

            dy.renew_cg()

            # Sample y_i ~ q(y|x) iid
            sampled_batch = [
                self.post_model.sample(get_unlabeled_sentence())[0]  # select only the tree
                for _ in range(self.num_samples)
            ]

            labeled_loss = dy.esum([self.joint_model.forward(tree) for tree in labeled_batch])
            sampled_loss = dy.esum([self.joint_model.forward(tree) for tree in sampled_batch])
            loss = (labeled_loss + sampled_loss) / (self.batch_size + self.num_samples)

            # Optimize objective
            loss.forward()
            loss.backward()
            self.optimizer.update()

            # Store losses
            self.wake_losses['total'].append(loss.value())
            self.wake_losses['labeled'].append(labeled_loss.value() / self.batch_size)
            self.wake_losses['sampled'].append(sampled_loss.value() / self.num_samples)

            self.num_wake_updates += 1

            if self.num_wake_updates % self.print_every == 0:
                loss = np.mean(self.wake_losses['total'][-self.print_every:])
                labeled_loss = np.mean(self.wake_losses['labeled'][-self.print_every:])
                sampled_loss = np.mean(self.wake_losses['sampled'][-self.print_every:])

                self.tensorboard_writer.add_scalar(
                    'wake-sleep/wake/loss', loss, self.num_wake_updates)
                self.tensorboard_writer.add_scalar(
                    'wake-sleep/wake/loss-labeled', labeled_loss, self.num_wake_updates)
                self.tensorboard_writer.add_scalar(
                    'wake-sleep/wake/loss-sampled', sampled_loss, self.num_wake_updates)

                print('| epoch {:2d} | wake phase | update {:4d}/{:5d} ({:.1%}) | loss {:6.3f} | elapsed {} | {:.2f} updates/sec | eta {}'.format(
                    self.current_epoch, i, len(labeled_batches), i / len(labeled_batches), loss,
                    self.timer.format_elapsed(), i / wake_timer.elapsed(),
                    self.timer.format(i * wake_timer.elapsed() / len(labeled_batches)),
                    self.timer.format((len(labeled_batches) - i) / i * wake_timer.elapsed())))

    def predict(self, examples):
        self.post_model.eval()
        trees = []
        for tree in tqdm(examples):
            dy.renew_cg()
            tree, *rest = self.post_model.parse(list(tree.leaves()))
            trees.append(tree.linearize())
        self.post_model.train()
        return trees

    def check_dev_fscore(self):
        print('Evaluating F1 on development set... (posterior model)')

        # Predict trees.
        trees = self.predict(self.dev_treebank)
        with open(self.dev_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)

        # Compute f-score.
        dev_fscore = evalb(
            self.evalb_dir, self.dev_pred_path, self.dev_path, self.dev_result_path)

        # Log score to tensorboard.
        self.current_dev_fscore = dev_fscore
        self.tensorboard_writer.add_scalar(
            'wake-sleep/dev-f-score', dev_fscore, self.num_wake_updates)

        return dev_fscore

    def check_dev_perplexity(self):
        print('Evaluating perplexity on development set... (posterior and joint model)')

        decoder = GenerativeDecoder(
            model=self.joint_model,
            proposal=self.post_model,
            num_samples=100,
            alpha=1.
        )

        pp = 0.
        for tree in tqdm(self.dev_treebank):
            dy.renew_cg()
            pp += decoder.perplexity(list(tree.leaves()))
        avg_pp = pp / len(self.dev_treebank)

        self.tensorboard_writer.add_scalar(
            'wake-sleep/dev-perplexity', avg_pp, self.num_wake_updates)

        return avg_pp


def blockgrad(expression):
    """Detach the expression from the computation graph"""
    if isinstance(expression, dy.Expression):
        return expression.value()
    else:  # already detached
        return expression
