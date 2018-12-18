import os
import json
import itertools
from math import inf
from collections import Counter

import numpy as np
import dynet as dy
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils.vocabulary import Vocabulary, UNK
from utils.trees import fromstring
from rnng.parser.actions import SHIFT, REDUCE, NT, GEN
from rnng.decoder import GenerativeDecoder
from rnng.model import DiscRNNG, GenRNNG
from rnng.components.feedforward import Feedforward, Affine
from utils.evalb import evalb
from utils.text import replace_quotes, replace_brackets
from utils.general import Timer, get_folders, write_args, ceil_div, blockgrad


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
        self.current_dev_fscore = -inf
        self.current_dev_perplexity = -inf

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

        print(f'Loading training trees from `{self.train_path}`...')
        with open(self.train_path) as f:
            train_treebank = [fromstring(line.strip()) for line in f]

        print(f'Loading development trees from `{self.dev_path}`...')
        with open(self.dev_path) as f:
            dev_treebank = [fromstring(line.strip()) for line in f]

        print(f'Loading test trees from `{self.test_path}`...')
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
        words = [word for tree in train_treebank for word in tree.words()] + [UNK]
        nonterminals = [label for tree in train_treebank for label in tree.labels()]

        word_vocab = Vocabulary.fromlist(words, unk_value=UNK)
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
            f'Vocab: {word_vocab.size:,} words, {nt_vocab.size:,} nonterminals, {gen_action_vocab.size:,} actions',
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

        try:
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
        except KeyboardInterrupt:
            print('-'*99)
            print('Exiting from training early.')
            print('-'*99)

        fscore = self.check_dev_fscore()
        pp = self.check_dev_perplexity()
        print(89*'=')
        print('| End of training | dev F1 {:4.2f} | dev perplexity {:4.2f}'.format(
            fscore, pp))
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
            tree, *rest = self.post_model.parse(tree.words())
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
            pp += decoder.perplexity(tree.words())
        avg_pp = pp / len(self.dev_treebank)

        self.current_dev_perplexity = dev_perplexity
        self.tensorboard_writer.add_scalar(
            'wake-sleep/dev-perplexity', avg_pp, self.num_wake_updates)

        return avg_pp

    def save_checkpoint(self):
        dy.save(self.post_model_checkpoint_path, [self.post_model])
        dy.save(self.joint_model_checkpoint_path, [self.joint_model])

        self.word_vocab.save(self.word_vocab_path)
        self.nt_vocab.save(self.nt_vocab_path)
        self.action_vocab.save(self.action_vocab_path)

        with open(self.joint_state_checkpoint_path, 'w') as f:
            state = {
                'rnng-type': 'gen',
                'semisup-method': 'wake-sleep',
                'epochs': self.current_epoch,
                'num-updates': self.num_updates,
                'dev-perplexity': self.current_dev_perplexity,
            }
            json.dump(state, f, indent=4)

        with open(self.post_state_checkpoint_path, 'w') as f:
            state = {
                'rnng-type': 'disc',
                'semisup-method': 'wake-sleep',
                'epochs': self.current_epoch,
                'num-updates': self.num_updates,
                'dev-fscore': self.current_dev_fscore,
            }
            json.dump(state, f, indent=4)
