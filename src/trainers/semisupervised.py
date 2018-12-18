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

        self.current_dev_fscore = -inf
        self.current_dev_perplexity = -inf

        self.test_fscore = -inf
        self.test_perplexity = -inf

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
            unlabeled_data = [
                replace_brackets(replace_quotes(line.strip().split()))
                for line in f
                if len(line.split()) < self.max_unlabeled_sent_len
            ]

        # TODO: Here we should load the vocabulary from the saved model!
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

                print('='*89)
                print(f'| End of epoch {epoch} | elapsed {self.timer.elapsed()}')
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
                self.check_dev()
                self.save_checkpoint()
                self.timer.new_epoch()
                print(89*'=')
                print('| dev F1 {:4.2f} | dev perplexity {:4.2f}'.format(
                    self.current_dev_fscore, self.current_dev_perplexity))
                print(89*'=')

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

    def estimate_elbo(self, n=2000):
        """Estimate the ELBO using the training samples."""
        # Recall that 1/n ELBO = 1/n E_q[log p(x, y) - q(y|x)] = 1/n learning_signal
        return np.mean(self.learning_signals[-n:])  # we take a running average for scaled comparison

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

    def check_dev(self):
        print('Evaluating F1 and perplexity on development set...')

        decoder = GenerativeDecoder(
            model=self.joint_model, proposal=self.post_model)

        print('Sampling proposals with posterior model...')
        decoder.generate_proposal_samples(
            examples=self.dev_treebank, path=self.dev_proposals_path)

        print('Scoring proposals with joint model...')
        trees, dev_perplexity = decoder.predict_from_proposal_samples(
            path=self.dev_proposals_path)

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
            model=self.joint_model, proposal=self.post_model)

        print('Sampling proposals with posterior model...')
        decoder.generate_proposal_samples(
            examples=self.test_treebank, path=self.test_proposals_path)

        print('Scoring proposals with joint model...')
        trees, test_perplexity = decoder.predict_from_proposal_samples(
            path=self.test_proposals_path)

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
        self.nt_vocab.save(self.nt_vocab_path)
        self.disc_action_vocab.save(self.action_vocab_path)

        with open(self.joint_state_checkpoint_path, 'w') as f:
            state = {
                'rnng-type': 'gen',
                'semisup-method': 'vi',
                'epochs': self.current_epoch,
                'num-updates': self.num_updates,
                'dev-perplexity': self.current_dev_perplexity,
            }
            json.dump(state, f, indent=4)

        with open(self.post_state_checkpoint_path, 'w') as f:
            state = {
                'rnng-type': 'disc',
                'semisup-method': 'vi',
                'epochs': self.current_epoch,
                'num-updates': self.num_updates,
                'dev-fscore': self.current_dev_fscore,
            }
            json.dump(state, f, indent=4)
