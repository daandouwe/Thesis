import os
import json
import itertools
from math import inf
from collections import Counter
import time

import numpy as np
import dynet as dy
from tensorboardX import SummaryWriter
from tqdm import tqdm

from rnng.parser.actions import SHIFT, REDUCE, NT, GEN
from rnng.model import DiscRNNG, GenRNNG
from rnng.decoder import GenerativeDecoder
from crf.model import ChartParser
from components.baseline import FeedforwardBaseline
from utils.trees import fromstring
from utils.evalb import evalb
from utils.general import Timer, get_folders, write_args, ceil_div, move_to_final_folder, blockgrad


class SemiSupervisedTrainer:

    def __init__(
            self,
            model_type=None,
            model_path_base=None,
            args=None,
            evalb_dir=None,
            unlabeled_path=None,
            train_path=None,
            dev_path=None,
            test_path=None,
            joint_model_path=None,
            post_model_path=None,
            lmbda=0,
            use_argmax_baseline=False,
            use_mlp_baseline=False,
            clip_learning_signal=None,
            exact_entropy=True,
            max_epochs=inf,
            max_time=inf,
            num_samples=3,
            alpha=None,
            batch_size=1,
            optimizer_type=None,
            lr=None,
            lr_decay=None,
            lr_decay_patience=None,
            weight_decay=None,
            max_grad_norm=None,
            glove_dir=None,
            print_every=1,
            eval_every=-1,  # default is every epoch (-1)
            eval_at_start=False,
            num_dev_samples=None,
            num_test_samples=None
    ):
        assert model_type in ('semisup-disc', 'semisup-crf'), model_type

        posterior_type = model_type.split('-')[-1]

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
        self.model_path_base = model_path_base
        self.posterior_type = posterior_type

        # Baselines
        self.lmbda = lmbda  # scaling coefficient for supervised objective
        self.use_argmax_baseline = use_argmax_baseline
        self.use_mlp_baseline = use_mlp_baseline
        self.clip_learning_signal = clip_learning_signal
        self.exact_entropy = exact_entropy

        # Training
        self.max_epochs = max_epochs
        self.max_time = max_time
        self.num_samples = num_samples
        self.alpha = alpha
        self.batch_size = batch_size
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_patience = lr_decay_patience
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.print_every = print_every
        self.eval_every = eval_every
        self.eval_at_start = eval_at_start
        self.num_dev_samples = num_dev_samples
        self.num_test_samples = num_test_samples
        self.max_crf_line_len = 15

        self.num_updates = 0
        self.learning_signals = []
        self.baseline_values = []
        self.centered_learning_signals = []
        self.losses = []
        self.sup_losses = []
        self.unsup_losses = []
        self.baseline_losses = []
        self.reset_histogram()

        self.current_dev_fscore = -inf
        self.current_dev_perplexity = inf

        self.test_fscore = -inf
        self.test_perplexity = inf

    def build_paths(self):
        # Make output folder structure
        subdir, logdir, checkdir, outdir, vocabdir = get_folders(self.args)

        joint_dir = os.path.join(checkdir, 'joint')
        post_dir = os.path.join(checkdir, 'posterior')

        os.makedirs(logdir, exist_ok=True)
        os.makedirs(checkdir, exist_ok=True)
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(vocabdir, exist_ok=True)
        os.makedirs(joint_dir, exist_ok=True)
        os.makedirs(post_dir, exist_ok=True)

        print(f'Output subdirectory: `{subdir}`.')
        print(f'Saving logs to `{logdir}`.')
        print(f'Saving predictions to `{outdir}`.')
        print(f'Saving models to `{checkdir}`.')

        # Save arguments
        write_args(self.args, logdir)

        self.subdir = subdir

        # Model paths
        self.post_model_checkpoint_path = os.path.join(post_dir, 'model')
        self.joint_model_checkpoint_path = os.path.join(joint_dir, 'model')
        self.post_state_checkpoint_path = os.path.join(post_dir, 'state.json')
        self.joint_state_checkpoint_path = os.path.join(joint_dir, 'state.json')

        self.word_vocab_path = os.path.join(checkdir, 'word-vocab.json')
        self.label_vocab_path = os.path.join(checkdir, 'label-vocab.json')
        self.action_vocab_path = os.path.join(checkdir, 'action-vocab.json')

        # Output paths
        self.loss_path = os.path.join(logdir, 'loss.csv')
        self.tensorboard_writer = SummaryWriter(logdir)

        # Dev paths
        self.dev_proposals_path = os.path.join(outdir, 'dev.pred.props')
        self.dev_pred_path = os.path.join(outdir, 'dev.pred.trees')
        self.dev_result_path = os.path.join(outdir, 'dev.result')

        # Test paths
        self.test_proposals_path = os.path.join(outdir, 'test.pred.props')
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
            unlabeled_data = [line.strip().split() for line in f]

        # NOTE: a very dirty solution to the speed problems of the crf...
        if self.posterior_type == 'crf':
            unlabeled_data = [line for line in unlabeled_data if len(line) < self.max_crf_line_len]

        self.word_vocab = self.joint_model.word_vocab
        self.label_vocab = self.joint_model.nt_vocab
        self.action_vocab = self.joint_model.action_vocab

        self.train_treebank = train_treebank
        self.dev_treebank = dev_treebank
        self.test_treebank = test_treebank
        self.unlabeled_data = unlabeled_data

        print('\n'.join((
            'Corpus statistics:',
            f'Vocab: {self.word_vocab.size:,} words, {self.label_vocab.size:,} nonterminals, {self.action_vocab.size:,} actions',
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

        epochs = state['num-epochs']
        fscore = state['best-dev-fscore']
        perplexity = state['best-dev-perplexity']
        print(f'Loaded joint model trained for {epochs} epochs with dev fscore {fscore} and perplexity {perplexity}.')

        self.tensorboard_writer.add_scalar(
            'semisup/dev/f-score', fscore, self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/dev/perplexity', perplexity, self.num_updates)

        self.current_dev_fscore = fscore
        self.current_dev_perplexity = perplexity

    def load_post_model(self):
        assert self.model is not None, 'build model first'

        model_path = os.path.join(self.post_model_path, 'model')
        state_path = os.path.join(self.post_model_path, 'state.json')

        [self.post_model] = dy.load(model_path, self.model)
        if self.posterior_type == 'disc':
            assert isinstance(self.post_model, DiscRNNG), type(self.post_model)
        elif self.posterior_type == 'crf':
            assert isinstance(self.post_model, ChartParser), type(self.post_model)
        self.post_model.train()

        with open(state_path) as f:
            state = json.load(f)
        epochs, fscore = state['num-epochs'], state['best-dev-fscore']
        print(f'Loaded posterior model of type `{self.posterior_type}` trained for {epochs} epochs with dev fscore {fscore}.')

    def build_optimizer(self):
        assert self.model is not None, 'build model first'

        if self.optimizer_type == 'sgd':
            self.optimizer = dy.SimpleSGDTrainer(self.model, learning_rate=self.lr)
            if self.use_mlp_baseline:
                self.baseline_optimizer = dy.SimpleSGDTrainer(
                    self.baseline_parameters, learning_rate=10*self.lr)

        elif self.optimizer_type == 'adam':
            self.optimizer = dy.AdamTrainer(self.model, alpha=self.lr)
            if self.use_mlp_baseline:
                self.baseline_optimizer = dy.AdamTrainer(
                    self.baseline_parameters, alpha=10*self.lr)

        self.model.set_weight_decay(self.weight_decay)
        self.optimizer.set_clip_threshold(self.max_grad_norm)
        if self.use_mlp_baseline:
            self.baseline_optimizer.set_clip_threshold(self.max_grad_norm)

    def build_baseline_model(self):
        self.baseline_parameters = dy.ParameterCollection()

        if self.use_mlp_baseline:
            print('Building feedforward baseline model...')

            if self.posterior_type == 'crf':
                lstm_dim = 2 * self.post_model.lstm_dim
            elif self.posterior_type == 'disc':
                lstm_dim = self.post_model.buffer_encoder.hidden_size

            self.baseline_model = FeedforwardBaseline(
                self.baseline_parameters, self.posterior_type, lstm_dim)

    def reset_histogram(self):
        self.histogram = dict(
            signal=[], baseline=[], centered=[], normalized=[],
            post=[], joint=[], unique=[], scale=[], entropy=[])

    def batchify(self, data):
        batches = [data[i*self.batch_size:(i+1)*self.batch_size]
            for i in range(ceil_div(len(data), self.batch_size))]
        return batches

    def train(self):
        self.build_paths()
        self.load_models()
        self.build_corpus()
        if self.use_mlp_baseline:
            self.build_baseline_model()
        self.build_optimizer()

        self.num_updates = 0
        self.unlabeled_batches = iter(self.batchify(self.unlabeled_data))

        self.timer = Timer()

        if self.eval_at_start:
            print('Evaluating at start...')

            self.check_dev()

            print(89*'=')
            print('| Start | dev F1 {:4.2f} | dev perplexity {:4.2f}'.format(
                self.current_dev_fscore, self.current_dev_perplexity))
            print(89*'=')

        try:
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
                self.save_checkpoint()
                print('='*89)
                print('| End of epoch {} | elapsed {} | dev F1 {:4.2f} | dev perplexity {:4.2f}'.format(
                    epoch, self.timer.format_elapsed(), self.current_dev_fscore, self.current_dev_perplexity))
                print('='*89)
        except KeyboardInterrupt:
            print('-'*99)
            print('Exiting from training early.')
            print('-'*99)

        self.check_dev()
        self.check_test()
        self.save_checkpoint()

        print(89*'=')
        print('| End of training | test F1 {:4.2f} | test perplexity {:4.2f}'.format(
            self.test_fscore, self.test_perplexity))
        print(89*'=')

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

            # print()
            # t0 = time.time()
            sup_loss = self.supervised_step(labeled_batch)
            # t1 = time.time()
            # print('supervised', t1 - t0)

            unsup_loss, baseline_loss = self.unsupervised_step(get_unlabeled_batch())
            # t2 = time.time()
            # print('unsupervised', t2 - t1)

            loss = sup_loss + unsup_loss

            # Optimize objective
            loss.forward()
            loss.backward()
            self.optimizer.update()
            # t3 = time.time()
            # print('loss update', t3 - t2)

            # Optimize baseline
            if self.use_mlp_baseline:
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

                self.write_tensoboard_histogram()
                self.write_tensoboard_losses(loss, sup_loss, unsup_loss, baseline_loss)
                self.reset_histogram()

                print('| epoch {:3d} | step {:5d}/{:5d} ({:.1%}) | sup-loss {:5.3f} | unsup-elbo {:.3f} | elapsed {} | {:.2f} updates/sec | eta {}'.format(
                    self.current_epoch, i, len(labeled_batches), i / len(labeled_batches),
                    sup_loss, self.estimate_elbo(),
                    self.timer.format_elapsed_epoch(), i / self.timer.elapsed_epoch(),
                    self.timer.format_eta(i, len(labeled_batches))))

            # if self.num_updates % self.eval_every == 0 and self.eval_every != -1:
            #     self.check_dev()
            #     self.save_checkpoint()
            #     self.timer.new_epoch()
            #     print(89*'=')
            #     print('| dev F1 {:4.2f} | dev perplexity {:4.2f}'.format(
            #         self.current_dev_fscore, self.current_dev_perplexity))
            #     print(89*'=')

    def supervised_step(self, batch):
        losses = []
        for tree in batch:
            post_tree = tree.cnf() if self.posterior_type == 'crf' else tree
            if self.lmbda == 0:
                loss = self.joint_model.forward(tree)
            else:
                loss = self.joint_model.forward(tree) + self.lmbda * self.post_model.forward(post_tree)
            losses.append(loss)
        loss = dy.esum(losses) / self.batch_size
        return loss

    def unsupervised_step(self, batch):
        if self.posterior_type == 'disc':
            return self.unsupervised_step_disc(batch)
        else:
            return self.unsupervised_step_crf(batch)

    def unsupervised_step_disc(self, batch):

        def sample(words):
            samples = [self.post_model.sample(words, self.alpha)
                for _ in range(self.num_samples)]
            trees, nlls = zip(*samples)
            logprob = dy.esum([-nll for nll in nlls]) / len(nlls)
            return trees, logprob

        def argmax_baseline(words):
            """Parameter-free baseline based on argmax decoding."""
            tree, post_nll = self.post_model.parse(words)
            post_logprob = -post_nll
            joint_logprob = -self.joint_model.forward(tree)
            return dy.scalarInput(blockgrad(joint_logprob - post_logprob))

        def baselines(words):
            b = dy.scalarInput(0)
            if self.use_mlp_baseline:
                b += self.mlp_baseline(words)
            if self.use_argmax_baseline:
                b += argmax_baseline(words)
            return b

        # Keep track of losses
        losses = []
        baseline_losses = []

        for words in batch:
            # Get samples with their \mean_y [log q(y|x)] for samples y
            trees, post_logprob = sample(words)

            # Compute mean \mean_y [log p(x,y)] for samples y
            joint_logprob = self.joint(trees)

            # Compute \mean_y [log p(x,y) - log q(y|x)] and detach
            learning_signal = blockgrad(joint_logprob - post_logprob)

            # Substract baseline
            baseline = baselines(words)
            a = self.optimal_baseline_scale()
            centered_learning_signal = learning_signal - a * baseline
            # centered_learning_signal = learning_signal - baseline

            # Normalize
            normalized_learning_signal = self.normalize(centered_learning_signal)

            # Optional clipping of learning signal
            normalized_learning_signal = self.clip(normalized_learning_signal)

            baseline_loss = centered_learning_signal**2
            post_loss = -blockgrad(normalized_learning_signal) * post_logprob
            joint_loss = -joint_logprob
            loss = post_loss + joint_loss

            losses.append(loss)
            baseline_losses.append(baseline_loss)

            # For tesorboard logging
            self.histogram['signal'].append(learning_signal)
            self.histogram['baseline'].append(baseline.value())
            self.histogram['scale'].append(a)
            self.histogram['centered'].append(centered_learning_signal.value())
            self.histogram['normalized'].append(normalized_learning_signal.value())
            self.histogram['post'].append(post_logprob.value())
            self.histogram['joint'].append(joint_logprob.value())
            # self.histogram['unique'].append(len(set(tree.linearize() for tree in trees)))

        # save the batch average
        self.learning_signals.append(
            np.mean(self.histogram['signal'][-self.batch_size:]))
        self.baseline_values.append(
            np.mean(self.histogram['baseline'][-self.batch_size:]))
        self.centered_learning_signals.append(
            np.mean(self.histogram['centered'][-self.batch_size:]))

        # Average losses over minibatch
        loss = dy.esum(losses) / self.batch_size
        baseline_loss = dy.esum(baseline_losses) / self.batch_size

        return loss, baseline_loss

    def unsupervised_step_crf(self, batch):

        def argmax_baseline(tree):
            """Parameter-free baseline based on argmax decoding."""
            joint_logprob = -self.joint_model.forward(tree)
            return dy.scalarInput(blockgrad(joint_logprob))

        def baselines(words, tree):
            b = dy.scalarInput(0)
            if self.use_mlp_baseline:
                b += self.mlp_baseline(words)
            if self.use_argmax_baseline:
                b += argmax_baseline(tree)
            return b

        # Keep track of losses
        losses = []
        baseline_losses = []

        for words in batch:
            if self.exact_entropy:
                # Combined computation is most efficient
                parse, samples, post_entropy = self.post_model.parse_sample_entropy(
                    words, self.num_samples)
            else:
                parse, samples = self.post_model.parse_sample(words, self.num_samples)

            # Compute \mean_y [log p(y|x)] for samples
            trees, post_nlls = zip(*samples)
            post_logprob = dy.esum([-nll for nll in post_nlls]) / len(post_nlls)

            # Compute mean \mean_y [log p(x,y)] for samples y
            joint_logprob = self.joint(trees)

            # Compute \mean_y [log p(x,y)]
            if self.exact_entropy:
                learning_signal = blockgrad(joint_logprob)
            else:
                learning_signal = blockgrad(joint_logprob - post_logprob)

            # Substract baseline
            baseline = baselines(words, parse)
            a = self.optimal_baseline_scale()
            centered_learning_signal = learning_signal - a * baseline

            # Normalize
            normalized_learning_signal = self.normalize(centered_learning_signal)

            # Optional clipping of learning signal
            normalized_learning_signal = self.clip(normalized_learning_signal)

            baseline_loss = centered_learning_signal**2
            if self.exact_entropy:
                post_loss = -(blockgrad(normalized_learning_signal) * post_logprob + post_entropy)
            else:
                post_loss = -blockgrad(normalized_learning_signal) * post_logprob
            joint_loss = -joint_logprob
            loss = post_loss + joint_loss

            losses.append(loss)
            baseline_losses.append(baseline_loss)

            # For tesorboard logging
            self.histogram['signal'].append(learning_signal)
            self.histogram['baseline'].append(baseline.value())
            self.histogram['scale'].append(a)
            self.histogram['centered'].append(centered_learning_signal.value())
            self.histogram['normalized'].append(normalized_learning_signal.value())
            self.histogram['post'].append(post_logprob.value())
            self.histogram['joint'].append(joint_logprob.value())
            if self.exact_entropy:
                self.histogram['entropy'].append(post_entropy.value())
            # self.histogram['unique'].append(len(set(tree.linearize() for tree in trees)))

        self.learning_signals.append(
            np.mean(self.histogram['signal'][-self.batch_size:]))
        self.baseline_values.append(
            np.mean(self.histogram['baseline'][-self.batch_size:]))
        self.centered_learning_signals.append(
            np.mean(self.histogram['centered'][-self.batch_size:]))

        # Average losses over minibatch
        loss = dy.esum(losses) / self.batch_size
        baseline_loss = dy.esum(baseline_losses) / self.batch_size

        return loss, baseline_loss

    # def posterior(self, trees):
    #     logprobs = [-self.post_model.forward(tree) for tree in trees]
    #     return dy.esum(logprobs) / len(logprobs)

    def joint(self, trees):
        logprobs = [-self.joint_model.forward(tree) for tree in trees]
        return dy.esum(logprobs) / len(logprobs)

    def mlp_baseline(self, words):
        """Baseline parametrized by a feedfoward network."""
        return self.baseline_model.forward(words, self.post_model)

    def optimal_baseline_scale(self, n=500):
        """Estimate optimal scaling for baseline."""
        if self.use_mlp_baseline:
            # baseline is adaptive so scaling messes it up
            return 1.
        elif self.num_updates < 4:
            # cannot yet estimate variance
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

    def clip(self, signal):
        if self.clip_learning_signal is not None:
            if signal.value() < self.clip_learning_signal:
                signal = dy.scalarInput(self.clip_learning_signal)
        return signal

    def estimate_elbo(self, n=2000):
        """Estimate the ELBO using the training samples.

        Recall that
            ELBO = E_q[log p(x, y) - q(y|x)]
                 = 1/n sum learning_signal
        """
        # take running average for batch-size independent comparison
        return np.mean(self.learning_signals[-n:])

    def write_tensoboard_losses(self, loss, sup_loss, unsup_loss, baseline_loss):
        self.tensorboard_writer.add_scalar(
            'semisup/loss/total', loss, self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/loss/sup', sup_loss, self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/loss/unsup-elbo', self.estimate_elbo(), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/loss/unsup-surrogate', unsup_loss, self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/loss/baseline', baseline_loss, self.num_updates)

    def write_tensoboard_histogram(self):

        # Write batch values as histograms
        self.tensorboard_writer.add_histogram(
            'semisup/histogram/learning-signal',
            np.array(self.histogram['signal']), self.num_updates)
        self.tensorboard_writer.add_histogram(
            'semisup/histogram/baseline',
            np.array(self.histogram['baseline']), self.num_updates)
        self.tensorboard_writer.add_histogram(
            'semisup/histogram/scale',
            np.array(self.histogram['scale']), self.num_updates)
        self.tensorboard_writer.add_histogram(
            'semisup/histogram/centered-learning-signal',
            np.array(self.histogram['centered']), self.num_updates)
        self.tensorboard_writer.add_histogram(
            'semisup/histogram/normalized-learning-signal',
            np.array(self.histogram['normalized']), self.num_updates)
        self.tensorboard_writer.add_histogram(
            'semisup/histogram/post-logprob',
            np.array(self.histogram['post']), self.num_updates)
        self.tensorboard_writer.add_histogram(
            'semisup/histogram/joint-logprob',
            np.array(self.histogram['joint']), self.num_updates)
        # self.tensorboard_writer.add_histogram(
            # 'semisup/histogram/unique-samples',
            # np.array(self.histogram['unique']), self.num_updates)

        # Write batch means as scalar
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/learning-signal',
            np.mean(self.histogram['signal']), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/baseline',
            np.mean(self.histogram['baseline']), self.num_updates)
        self.tensorboard_writer.add_histogram(
            'semisup/unsup/scale',
            np.mean(self.histogram['scale']), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/centered-learning-signal',
            np.mean(self.histogram['centered']), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/normalized-learning-signal',
            np.mean(self.histogram['normalized']), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/post-logprob',
            np.mean(self.histogram['post']), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/joint-logprob',
            np.mean(self.histogram['joint']), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/joint-logprob',
            np.mean(self.histogram['joint']), self.num_updates)
        if self.posterior_type == 'crf' and self.exact_entropy:
            self.tensorboard_writer.add_scalar(
                'semisup/unsup/post-entropy',
                np.mean(self.histogram['entropy']), self.num_updates)

        # Write signal statistics
        cov, corr = self.baseline_signal_covariance()
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/signal-mean',
            np.mean(self.learning_signals[-100:]), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/signal-variance',
            np.var(self.learning_signals[-100:]), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/centered-signal-mean',
            np.mean(self.centered_learning_signals[-100:]), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/centered-signal-variance',
            np.var(self.centered_learning_signals[-100:]), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/baseline-mean',
            np.mean(self.baseline_values[-100:]), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/baseline-variance',
            np.var(self.baseline_values[-100:]), self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/signal-baseline-cov',
            cov, self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/unsup/signal-baseline-cor',
            corr, self.num_updates)
        self.tensorboard_writer.add_scalar(  # var(f') / var(f) = 1 - corr(f, b)**2
            'semisup/unsup/signal-baseline-var-frac',
            1 - corr**2, self.num_updates)

    def baseline_signal_covariance(self, n=200):
        baseline_values = np.array(self.baseline_values[-n:])
        signal_values = np.array(self.learning_signals[-n:])

        signal_var, cov, _, baseline_var = np.cov(
            [signal_values, baseline_values]).ravel()

        # baseline_mean = np.mean(baseline_values)
        # signal_mean = np.mean(signal_values)
        # cov = np.mean((baseline_values - baseline_mean) * (signal_values - signal_mean))

        # baseline_std = np.std(baseline_values)
        # signal_std = np.std(signal_values)

        corr = cov / np.sqrt(baseline_var * signal_var)
        return cov, corr

    def check_dev(self):
        print('Evaluating F1 and perplexity on development set...')

        decoder = GenerativeDecoder(
            model=self.joint_model, proposal=self.post_model, num_samples=self.num_dev_samples)

        print('Sampling proposals with posterior model...')
        dev_sentences = [tree.words() for tree in self.dev_treebank]
        decoder.generate_proposal_samples(
            sentences=dev_sentences, outpath=self.dev_proposals_path)

        print('Scoring proposals with joint model...')
        trees, dev_perplexity = decoder.predict_from_proposal_samples(
            inpath=self.dev_proposals_path)

        with open(self.dev_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)

        dev_fscore = evalb(
            self.evalb_dir, self.dev_pred_path, self.dev_path, self.dev_result_path)

        # Log score to tensorboard
        self.tensorboard_writer.add_scalar(
            'semisup/dev/f-score', dev_fscore, self.num_updates)
        self.tensorboard_writer.add_scalar(
            'semisup/dev/perplexity', dev_perplexity, self.num_updates)

        self.current_dev_fscore = dev_fscore
        self.current_dev_perplexity = dev_perplexity

    def check_test(self):
        print('Evaluating F1 and perplexity on test set...')

        decoder = GenerativeDecoder(
            model=self.joint_model, proposal=self.post_model, num_samples=self.num_test_samples)

        print('Sampling proposals with posterior model...')
        test_sentences = [tree.words() for tree in self.test_treebank]
        decoder.generate_proposal_samples(
            sentences=test_sentences, outpath=self.test_proposals_path)

        print('Scoring proposals with joint model...')
        trees, test_perplexity = decoder.predict_from_proposal_samples(
            inpath=self.test_proposals_path)

        with open(self.test_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)

        test_fscore = evalb(
            self.evalb_dir, self.test_pred_path, self.test_path, self.test_result_path)

        # Log score to tensorboard
        self.tensorboard_writer.add_scalar(
            'semisup/test/f-score', test_fscore)
        self.tensorboard_writer.add_scalar(
            'semisup/test/perplexity', test_perplexity)

        self.test_fscore = test_fscore
        self.test_perplexity = test_perplexity

    def save_checkpoint(self):
        dy.save(self.post_model_checkpoint_path, [self.post_model])
        dy.save(self.joint_model_checkpoint_path, [self.joint_model])

        self.word_vocab.save(self.word_vocab_path)
        self.label_vocab.save(self.label_vocab_path)
        self.action_vocab.save(self.action_vocab_path)

        with open(self.joint_state_checkpoint_path, 'w') as f:
            state = {
                'model': 'gen-rnng',

                'num-epochs': self.current_epoch,
                'num-updates': self.num_updates,
                'elapsed': self.timer.format_elapsed(),

                'current-dev-fscore': self.current_dev_fscore,
                'best-dev-fscore': self.current_dev_fscore,
                'test-fscore': self.test_fscore,

                'current-dev-perplexity': self.current_dev_perplexity,
                'best-dev-perplexity': self.current_dev_perplexity,
                'test-perplexity': self.test_perplexity,
            }
            json.dump(state, f, indent=4)

        with open(self.post_state_checkpoint_path, 'w') as f:
            state = {
                'model': self.posterior_type,

                'num-epochs': self.current_epoch,
                'num-updates': self.num_updates,
                'elapsed': self.timer.format_elapsed(),

                'best-dev-fscore': self.current_dev_fscore,
                'test-fscore': self.test_fscore,

                'best-dev-perplexity': self.current_dev_perplexity,
                'test-perplexity': self.test_perplexity,
            }
            json.dump(state, f, indent=4)

    def finalize_model_folder(self):
        move_to_final_folder(
            self.subdir, self.model_path_base, self.current_dev_perplexity)
