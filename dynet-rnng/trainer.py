import os
import json
import itertools
import time
import pickle
from math import inf

import numpy as np
import dynet as dy
from tensorboardX import SummaryWriter

from vocabulary import Vocabulary, UNK
from actions import SHIFT, REDUCE, NT, GEN
from tree import fromstring
from decode import GenerativeDecoder
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
            train_trees = [fromstring(line.strip()) for line in f]

        print(f'Loading development trees from `{self.dev_path}`...')
        with open(self.dev_path) as f:
            dev_trees = [fromstring(line.strip()) for line in f]

        print(f'Loading test trees from `{self.dev_path}`...')
        with open(self.test_path) as f:
            test_trees = [fromstring(line.strip()) for line in f]

        print("Constructing vocabularies...")
        words = [word for tree in train_trees for word in tree.leaves()] + [UNK]
        tags = [tag for tree in train_trees for tag in tree.tags()]
        nonterminals = [label for tree in train_trees for label in tree.labels()]

        word_vocab = Vocabulary.fromlist(words, unk=True)
        tag_vocab = Vocabulary.fromlist(tags)
        nt_vocab = Vocabulary.fromlist(nonterminals)

        # The order is very important!
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

        self.train_trees = train_trees
        self.dev_trees = dev_trees
        self.test_trees = test_trees

        print('\n'.join((
            'Corpus statistics:',
            f'Vocab: {word_vocab.size:,} words ({len(word_vocab.unks)} UNK-types), {nt_vocab.size:,} nonterminals, {action_vocab.size:,} actions',
            f'Train: {len(train_trees):,} sentences',
            f'Dev: {len(dev_trees):,} sentences',
            f'Test: {len(test_trees):,} sentences')))

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

        print('Start training...')
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
            np.random.shuffle(self.train_trees)
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
        num_sentences = len(self.train_trees)
        num_batches = num_sentences // self.batch_size
        processed = 0
        batches = self.batchify(self.train_trees)
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
        dev_trees = self.dev_trees[:30] if self.rnng_type == 'gen' else self.dev_trees # Is slooow!
        trees = self.predict(dev_trees, proposal_samples=self.dev_proposal_samples)
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
        trees = self.predict(self.test_trees, proposal_samples=self.test_proposal_samples)
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


class SemiSupervisedTrainer(Trainer):

    def __init__(
            self,
            args=None,
            name=None,
            data_dir=None,
            evalb_dir=None,
            lm_path=None,
            train_path=None,
            dev_path=None,
            test_path=None,
            text_type=None,
            joint_model_path=None,
            post_model_path=None,
            use_argmax_baseline=False,
            use_mean_baseline=False,
            use_lm_baseline=False,
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
        self.name = name
        self.data_dir = data_dir
        self.evalb_dir = evalb_dir
        self.lm_path = lm_path
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.text_type = text_type

        # Model paths
        self.joint_model_path = joint_model_path
        self.post_model_path = post_model_path

        # Baselines
        self.use_argmax_baseline = use_argmax_baseline
        self.use_mean_baseline = use_mean_baseline
        self.use_lm_baseline = use_lm_baseline

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

        self.cum_learning_signal = 0
        self.num_unsup_updates = 0
        self.unsup_losses = []

    def word_ids(self, words):
        return [self.gen_vocab.w2i[word] for word in words]

    def gen_actions_ids(self, actions):
        return [self.gen_vocab.a2i[action] for action in actions]

    def disc_actions_ids(self, actions):
        return [self.disc_vocab.a2i[action] for action in actions]

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
        corpus = SemiSupervisedCorpus(
            self.train_path,
            self.dev_path,
            self.test_path,
            self.lm_path,
            self.text_type
        )
        self.gen_vocab = corpus.gen_vocab
        self.disc_vocab = corpus.disc_vocab
        self.lm_dataset = corpus.lm.data
        self.train_trees = corpus.train.data
        self.dev_trees = corpus.dev.data
        self.test_trees = corpus.test.data
        print('\n'.join((
            'Corpus statistics:',
            f'Vocab {self.gen_vocab.num_words:,} words',
            f'LM {len(self.lm_dataset):,} sentences',
            f'Train {len(self.train_trees):,} sentences',
            f'Dev {len(self.dev_trees):,} sentences',
            f'Test {len(self.test_trees):,} sentences')))

    def load_joint_model(self):
        model_path = os.path.join(self.joint_model_path, 'model')
        state_path = os.path.join(self.joint_model_path, 'state.json')
        self.model = dy.ParameterCollection()
        [self.joint_model] = dy.load(model_path, self.model)
        assert isinstance(self.joint_model, GenRNNG), type(self.joint_model)
        self.joint_model.train()
        with open(state_path) as f:
            state = json.load(f)
        epochs, fscore = state['epochs'], state['test-fscore']
        print(f'Loaded joint model trained for {epochs} epochs with test fscore {fscore}.')

    def load_post_model(self):
        model_path = os.path.join(self.post_model_path, 'model')
        state_path = os.path.join(self.post_model_path, 'state.json')
        self.model = dy.ParameterCollection()
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

    def build_baseline_parameters(self):
        self.a_arg = self.post_model.model.add_parameters(1, init=0)
        self.c_arg = self.post_model.model.add_parameters(1, init=0)
        self.a_lm = self.post_model.model.add_parameters(1, init=0)
        self.c_lm = self.post_model.model.add_parameters(1, init=0)

    def train(self):
        self.build_paths()
        self.build_corpus()
        self.load_joint_model()
        self.load_post_model()
        self.build_baseline_parameters()
        self.build_optimizer()

        self.cum_learning_signal = 0
        self.num_unsup_updates = 0

        batches = self.batchify(self.lm_dataset)
        for batch in batches:
            self.unsupervised_step(batch)
            # self.sup_step(batch)

            if self.num_unsup_updates % self.eval_every == 0:
                fscore = self.check_dev()
                print('='*89)
                print(f'| Dev fscore {fscore:.2f} |')
                print('='*89)
            if self.num_unsup_updates % self.print_every == 0:
                avg_loss = np.mean(self.unsup_losses[-self.print_every:])
                self.tensorboard_writer.add_scalar(
                    'train/loss', avg_loss, self.num_unsup_updates)
                print(f'| Step {self.num_unsup_updates} | loss {avg_loss:.3f} |')

    def sample(self, words):
        samples = [self.post_model.sample(words, self.alpha) for _ in range(self.num_samples)]
        trees, nlls = zip(*samples)
        return trees, nlls

    def joint(self, trees):
        logprobs = [-self.joint_model(
            self.word_ids(tree.leaves()), self.gen_actions_ids(tree.gen_oracle())) for tree in trees]
        return dy.esum(logprobs) / len(logprobs)

    def posterior(self, trees):
        logprobs = [-self.post_model(
            self.word_ids(tree.leaves()), self.disc_actions_ids(tree.disc_oracle())) for tree in trees]
        return dy.esum(logprobs) / len(logprobs)

    def unsupervised_step(self, batch):
        dy.renew_cg()
        losses = []
        cum_learning_signal = 0
        for words in batch:
            # Get samples
            trees, _ = self.sample(words)

            # Compute mean mean_y log p(x,y) and mean mean_y log q(y|x)
            joint_logprob = self.joint(trees)
            post_logprob = self.posterior(trees)

            # Compute mean_y [log p(x,y) - log q(y|x)]
            learning_signal = blockgrad(joint_logprob - post_logprob)

            # Compute baseline
            baselines = self.baselines(words)
            centered_learning_signal = (learning_signal - baselines)

            post_loss = -centered_learning_signal * post_logprob
            joint_loss = -joint_logprob
            baseline_loss = centered_learning_signal**2

            # loss = post_loss + joint_loss + baseline_loss
            loss = post_loss + baseline_loss
            # loss = post_loss
            losses.append(loss)

            cum_learning_signal += learning_signal

            # Log to tensorboard
            self.tensorboard_writer.add_scalar(
                'train/learning-signal', learning_signal, self.num_unsup_updates)
            self.tensorboard_writer.add_scalar(
                'train/centered-learning-signal', centered_learning_signal.value(), self.num_unsup_updates)
            self.tensorboard_writer.add_scalar(
                'train/post-logprob', post_logprob.value(), self.num_unsup_updates)
            self.tensorboard_writer.add_scalar(
                'train/joint-logprob', joint_logprob.value(), self.num_unsup_updates)

        loss = dy.esum(losses) / self.batch_size
        # Update parameters
        loss.forward()
        loss.backward()
        self.optimizer.update()

        self.unsup_losses.append(loss.value())
        self.num_unsup_updates += 1
        self.cum_learning_signal += (cum_learning_signal / self.batch_size)

        if self.num_unsup_updates % self.print_every == 0:
            for tree in trees:
                print(tree.linearize(with_tag=False))

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
        if self.num_unsup_updates == 0:
            return 0
        else:
            baseline = self.cum_learning_signal / self.num_unsup_updates
            # Log to tensorboard
            self.tensorboard_writer.add_scalar(
                'train/mean-baseline', baseline, self.num_unsup_updates)
            return mean_baseline

    def lm_baseline(self, words):
        return self.a_lm * blockgrad(self.baseline_lm(words)) + self.c_lm

    def argmax_baseline(self, words):
        tree, _ = self.post_model.parse(words)
        actions = self.gen_actions_ids(tree.gen_oracle())
        joint_logprob = -self.joint_model(words, actions)
        baseline = self.a_arg * blockgrad(joint_logprob) + self.c_arg
        # Log to tensorboard
        self.tensorboard_writer.add_scalar(
            'train/argmax-baseline', baseline.value(), self.num_unsup_updates)
        self.tensorboard_writer.add_scalar(
            'train/argmax-logprob', joint_logprob.value(), self.num_unsup_updates)
        self.tensorboard_writer.add_scalar(
            'train/a_arg', self.a_arg.value(), self.num_unsup_updates)
        self.tensorboard_writer.add_scalar(
            'train/c_arg', self.c_arg.value(), self.num_unsup_updates)
        return baseline

    def predict(self, examples):
        self.post_model.eval()
        trees = []
        for i, (sentence, _) in enumerate(examples):
            dy.renew_cg()
            tree, *rest = self.post_model.parse(sentence)
            trees.append(tree.linearize())
            if i % 10 == 0:
                print(f'Predicting sentence {i}/{len(examples)}...', end='\r')
        self.post_model.train()
        return trees

    def check_dev(self):
        print('Evaluating F1 on development set...')
        # Predict trees.
        trees = self.predict(self.dev_trees)
        with open(self.dev_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)
        # Compute f-score.
        dev_fscore = evalb(
            self.evalb_dir, self.dev_pred_path, self.dev_gold_path, self.dev_result_path)
        # Log score to tensorboard.
        self.current_dev_fscore = dev_fscore
        self.tensorboard_writer.add_scalar('dev/f-score', dev_fscore, self.num_unsup_updates)
        return dev_fscore


def blockgrad(expression):
    """Detach the expression from the computation graph"""
    return expression.value()
