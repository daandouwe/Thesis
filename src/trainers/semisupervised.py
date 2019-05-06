import os
import json
import itertools
from math import inf
from collections import Counter

import numpy as np
import dynet as dy
from tensorboardX import SummaryWriter
from tqdm import tqdm

from rnng.parser.actions import SHIFT, REDUCE, NT, GEN
from rnng.model import DiscRNNG, GenRNNG
from rnng.decoder import GenerativeDecoder
from crf.model import ChartParser, START, STOP
from components.baseline import FeedforwardBaseline
from utils.vocabulary import Vocabulary, UNK
from utils.trees import fromstring, DUMMY, UNLABEL
from utils.evalb import evalb
from utils.general import Timer, get_folders, write_args, ceil_div, move_to_final_folder, blockgrad, is_tree


class SemiSupervisedTrainer:

    def __init__(
            self,
            model_type=None,
            model_path_base=None,
            args=None,
            evalb_dir=None,
            evalb_param_file=None,
            unlabeled_path=None,
            train_path=None,
            dev_path=None,
            test_path=None,
            vocab_path=None,
            unlabeled=None,
            joint_model_path=None,
            post_model_path=None,
            lmbda=0,
            max_crf_line_len=-1,
            use_argmax_baseline=False,
            use_mlp_baseline=False,
            normalize_learning_signal=False,
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
            anneal_entropy=False,
            num_anneal_epochs=2,
            weight_decay=None,
            max_grad_norm=None,
            glove_dir=None,
            print_every=1,
            eval_every=-1,  # default is every epoch (-1)
            eval_at_start=False,
            num_dev_samples=None,
            num_test_samples=None
    ):
        assert model_type in ('semisup-disc', 'semisup-crf', 'unsup-disc', 'unsup-crf'), model_type

        train_objective, posterior_type = model_type.split('-')

        if not unlabeled:
            evalb_param_file = None

        self.args = args

        # Data
        self.evalb_dir = evalb_dir
        self.evalb_param_file = evalb_param_file
        self.unlabeled_path = os.path.expanduser(unlabeled_path)
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.vocab_path = vocab_path
        self.unlabeled = unlabeled

        # Model paths
        self.model = None  # will be a dynet ParameterCollection
        self.load_pretrained = joint_model_path is not None
        self.joint_model_path = joint_model_path
        self.post_model_path = post_model_path
        self.model_path_base = model_path_base
        self.posterior_type = posterior_type

        # Baselines
        self.lmbda = lmbda  # scaling coefficient for supervised objective
        self.use_argmax_baseline = use_argmax_baseline
        self.use_mlp_baseline = use_mlp_baseline
        self.normalize_learning_signal = normalize_learning_signal
        self.clip_learning_signal = clip_learning_signal
        self.exact_entropy = exact_entropy

        # Training
        self.train_objective = train_objective
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
        self.max_crf_line_len = max_crf_line_len
        self.anneal_entropy = anneal_entropy
        self.num_anneal_epochs = num_anneal_epochs

        self.num_updates = 0
        self.build_logger()

        self.dev_joint_fscore = -inf
        self.dev_post_fscore = -inf
        self.dev_perplexity = inf

        self.test_joint_fscore = -inf
        self.test_post_fscore = -inf
        self.test_perplexity = inf

    def build_paths(self):
        # Make output folder structure
        subdir, logdir, checkdir, outdir, vocabdir = get_folders(self.args)

        joint_dir = os.path.join(checkdir, 'joint')
        post_dir = os.path.join(checkdir, 'posterior')

        os.makedirs(logdir, exist_ok=True)
        os.makedirs(checkdir, exist_ok=True)
        os.makedirs(vocabdir, exist_ok=True)
        os.makedirs(joint_dir, exist_ok=True)
        os.makedirs(post_dir, exist_ok=True)

        print(f'Output subdirectory: `{subdir}`.')
        print(f'Saving logs to `{logdir}`.')
        print(f'Saving joint model and predictions to `{joint_dir}`.')
        print(f'Saving posterior model and predictions to `{post_dir}`.')

        # Save arguments
        write_args(self.args, logdir)

        self.subdir = subdir

        # Model paths
        self.post_model_checkpoint_path = os.path.join(post_dir, 'model')
        self.joint_model_checkpoint_path = os.path.join(joint_dir, 'model')
        self.post_state_checkpoint_path = os.path.join(post_dir, 'state.json')
        self.joint_state_checkpoint_path = os.path.join(joint_dir, 'state.json')

        self.word_vocab_path = os.path.join(vocabdir, 'word-vocab.json')
        self.label_vocab_path = os.path.join(vocabdir, 'label-vocab.json')
        self.action_vocab_path = os.path.join(vocabdir, 'action-vocab.json')

        # Output paths
        self.loss_path = os.path.join(logdir, 'loss.csv')
        self.tensorboard_writer = SummaryWriter(logdir)

        # Dev paths
        self.dev_proposals_path = os.path.join(joint_dir, 'dev.pred.props')
        self.dev_joint_pred_path = os.path.join(joint_dir, 'dev.pred.trees')
        self.dev_joint_result_path = os.path.join(joint_dir, 'dev.result')

        self.dev_post_pred_path = os.path.join(post_dir, 'dev.pred.trees')
        self.dev_post_result_path = os.path.join(post_dir, 'dev.result')

        # Test paths
        self.test_proposals_path = os.path.join(joint_dir, 'test.pred.props')
        self.test_joint_pred_path = os.path.join(joint_dir, 'test.pred.trees')
        self.test_joint_result_path = os.path.join(joint_dir, 'test.result')

        self.test_post_pred_path = os.path.join(post_dir, 'test.pred.trees')
        self.test_post_result_path = os.path.join(post_dir, 'test.result')

    def build_corpus(self):
        # Building data
        print(f'Loading training data from `{self.train_path}`...')
        with open(self.train_path) as f:
            train_treebank = [fromstring(line.strip()) for line in f]

        print(f'Loading development trees from `{self.dev_path}`...')
        with open(self.dev_path) as f:
            dev_treebank = [fromstring(line.strip()) for line in f]

        print(f'Loading test trees from `{self.test_path}`...')
        with open(self.test_path) as f:
            test_treebank = [fromstring(line.strip()) for line in f]

        if self.posterior_type == 'crf' and self.max_crf_line_len > 0:
            print(f'Filtering training trees by length <= {self.max_crf_line_len}.')
            train_treebank = [tree for tree in train_treebank if len(tree.words()) <= self.max_crf_line_len]

        if self.unlabeled:
            print(f'Converting training trees to unlabeled form...')
            for tree in train_treebank:
                tree.unlabelize()

        self.train_treebank = train_treebank
        self.dev_treebank = dev_treebank
        self.test_treebank = test_treebank

        if self.train_objective == 'semisup':
            print(f'Loading unlabeled data from `{self.unlabeled_path}`...')
            with open(self.unlabeled_path) as f:
                lines = [line.strip() for line in f]

            if is_tree(lines[0]):
                unlabeled_data = [fromstring(line).words() for line in lines]
            else:
                unlabeled_data = [line.split() for line in lines]

            if self.posterior_type == 'crf' and self.max_crf_line_len > 0:
                unlabeled_data = [words for words in unlabeled_data if len(words) <= self.max_crf_line_len]

            self.unlabeled_data = unlabeled_data

        # Building vocabularies
        if self.load_pretrained:
            self.word_vocab = self.joint_model.word_vocab
            self.label_vocab = self.joint_model.nt_vocab
            self.action_vocab = self.joint_model.action_vocab
        else:
            print('Constructing vocabularies...')
            if self.vocab_path is None:
                words = [word for tree in train_treebank for word in tree.words()]
            else:
                print(f'Using word vocabulary specified in `{self.vocab_path}`')
                with open(self.vocab_path) as f:
                    vocab = json.load(f)
                words = [word for word, count in vocab.items() for _ in range(count)]

            if self.posterior_type == 'crf':
                crf_words = [UNK, START, STOP] + words
                crf_labels = [(DUMMY,), (UNLABEL,)]
                self.crf_word_vocab = Vocabulary.fromlist(crf_words, unk_value=UNK)
                self.crf_label_vocab = Vocabulary.fromlist(crf_labels)

            words = [UNK] + words
            labels = [label for tree in train_treebank for label in tree.labels()]

            self.word_vocab = Vocabulary.fromlist(words, unk_value=UNK)
            self.label_vocab = Vocabulary.fromlist(labels)

        if not self.load_pretrained:
            # Order is very important, see DiscParser class
            disc_actions = [SHIFT, REDUCE] + [NT(label) for label in self.label_vocab]
            disc_action_vocab = Vocabulary()
            for action in disc_actions:
                disc_action_vocab.add(action)

            # Order is very important, see GenParser class
            gen_action_vocab = Vocabulary()
            gen_actions = [REDUCE] + [NT(label) for label in self.label_vocab] + [GEN(word) for word in self.word_vocab]
            for action in gen_actions:
                gen_action_vocab.add(action)

            # Needed to build the RNNG models
            self.action_vocab = gen_action_vocab
            self.gen_action_vocab = gen_action_vocab
            self.disc_action_vocab = disc_action_vocab

        print('\n'.join((
            'Corpus statistics:',
            f'Vocab: {self.word_vocab.size:,} words, {self.label_vocab.size:,} nonterminals, {self.action_vocab.size:,} actions',
            f'Train: {len(self.train_treebank):,} sentences',
            f'Dev: {len(self.dev_treebank):,} sentences',
            f'Test: {len(self.test_treebank):,} sentences')))
        if self.train_objective == 'semisup':
            print(f'Unlabeled: {len(self.unlabeled_data):,} sentences')

    def build_models(self):
        assert self.word_vocab is not None, 'build corpus first'

        print('Building models...')
        self.model = dy.ParameterCollection()

        if self.posterior_type == 'crf':
            self.post_model = ChartParser(
                model=self.model,
                word_vocab=self.crf_word_vocab,
                label_vocab=self.crf_label_vocab,
                word_embedding_dim=100,
                lstm_layers=2,
                lstm_dim=128,
                label_hidden_dim=256,
                dropout=0.2,
            )
        else:
            self.post_model = DiscRNNG(
                model=self.model,
                word_vocab=self.word_vocab,
                nt_vocab=self.label_vocab,
                action_vocab=self.disc_action_vocab,
                word_emb_dim=100,
                nt_emb_dim=100,
                action_emb_dim=100,
                stack_lstm_dim=128,
                buffer_lstm_dim=128,
                history_lstm_dim=128,
                lstm_layers=2,
                composition='attention',
                f_hidden_dim=256,
                dropout=0.2,
            )
        self.joint_model = GenRNNG(
            model=self.model,
            word_vocab=self.word_vocab,
            nt_vocab=self.label_vocab,
            action_vocab=self.gen_action_vocab,
            word_emb_dim=100,
            nt_emb_dim=100,
            action_emb_dim=100,
            stack_lstm_dim=256,
            terminal_lstm_dim=256,
            history_lstm_dim=256,
            lstm_layers=2,
            composition='attention',
            f_hidden_dim=256,
            dropout=0.3,
        )

    def load_models(self):
        print('Loading pre-trained models...')
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
        dev_fscore = state['best-dev-fscore']
        dev_perplexity = state['best-dev-perplexity']
        test_fscore = state['test-fscore']
        test_perplexity = state['test-perplexity']
        print(f'Loaded joint model trained for {epochs} epochs with dev fscore {dev_fscore} and perplexity {dev_perplexity}.')

        self.tensorboard_writer.add_scalar(
            'dev/joint-f-score', dev_fscore, self.num_updates)
        self.tensorboard_writer.add_scalar(
            'dev/perplexity', dev_perplexity, self.num_updates)
        self.tensorboard_writer.add_scalar(
            'test/joint-f-score', test_fscore, self.num_updates)
        self.tensorboard_writer.add_scalar(
            'test/perplexity', test_perplexity, self.num_updates)

        self.dev_joint_fscore = dev_fscore
        self.dev_perplexity = dev_perplexity
        self.test_joint_fscore = test_fscore
        self.test_perplexity = test_perplexity

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

        epochs = state['num-epochs']
        dev_fscore = state['best-dev-fscore']
        test_fscore = state['test-fscore']

        print(f'Loaded posterior model of type `{self.posterior_type}` trained for {epochs} epochs with dev fscore {dev_fscore}.')

        self.tensorboard_writer.add_scalar(
            'dev/post-f-score', dev_fscore, self.num_updates)
        self.tensorboard_writer.add_scalar(
            'test/post-f-score', test_fscore, self.num_updates)

        self.dev_post_fscore = dev_fscore
        self.test_post_fscore = test_fscore

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

    def build_logger(self):
        self.logger = {
            'unsup-loss': [],
            'sup-loss': [],
            'loss': [],
            'signal': [],
            'baseline': [],
            'centered': [],
            'scale': [],
            'entropy': [],
            'post': [],
            'joint': [],
            'post-viterbi': [],
            'anneal-entropy': [],
            'unique-samples': [],
        }

    def batchify(self, data):
        batches = [data[i*self.batch_size:(i+1)*self.batch_size]
            for i in range(ceil_div(len(data), self.batch_size))]
        return batches

    def anneal(self):
        """Linear annealing spread out over epochs."""
        if self.current_epoch > self.num_anneal_epochs:
            return 1.
        else:
            return self.num_updates / (self.num_anneal_epochs * self.num_batches)

    def train(self):
        self.build_paths()
        if self.load_pretrained:
            self.load_models()
            self.build_corpus()
        else:
            self.build_corpus()
            self.build_models()

        if self.use_mlp_baseline:
            self.build_baseline_model()
        self.build_optimizer()

        if self.train_objective == 'semisup':
            self.unlabeled_batches = iter(self.batchify(self.unlabeled_data))

        self.num_updates = 0
        self.timer = Timer()

        if self.eval_at_start:
            print('Evaluating at start...')

            self.check_dev()

            print(89*'=')
            print('| Start | dev F1 {:4.2f} | dev perplexity {:4.2f}'.format(
                self.dev_joint_fscore, self.dev_perplexity))
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
                if self.train_objective == 'semisup':
                    self.train_epoch_semisup()
                else:
                    self.train_epoch_unsup()

                # Check development scores
                self.check_dev()
                self.save_checkpoint()
                print('='*89)
                print('| End of epoch {} | elapsed {} | dev F1 {:4.2f} | dev perplexity {:4.2f}'.format(
                    epoch, self.timer.format_elapsed(), self.dev_joint_fscore, self.dev_perplexity))
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
            self.test_joint_fscore, self.test_perplexity))
        print(89*'=')

    def train_epoch_semisup(self):

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
        self.num_batches = len(labeled_batches)

        # We loop over the labeled_batches and request an unlabeled batch
        for i, labeled_batch in enumerate(labeled_batches, 1):

            if self.timer.elapsed() > self.max_time:
                break

            dy.renew_cg()

            sup_loss = self.supervised_step(labeled_batch)
            unsup_loss = self.unsupervised_step(get_unlabeled_batch())
            loss = sup_loss + unsup_loss

            # Optimize objective
            loss.forward()
            loss.backward()
            self.optimizer.update()

            # Store losses
            self.logger['sup-loss'].append(sup_loss.value())
            self.logger['unsup-loss'].append(unsup_loss.value())
            self.logger['loss'].append(loss.value())

            self.num_updates += 1

            if i % self.print_every == 0:
                self.write_tensoboard()
                sup_loss = np.mean(self.logger['sup-loss'][-self.print_every:])

                print('| epoch {:3d} | step {:5d}/{:5d} ({:.1%}) | sup-loss {:8.3f} | unsup-elbo {:8.3f} | elapsed {} | {:3.2f} updates/sec | eta {}'.format(
                    self.current_epoch, i, len(labeled_batches), i / len(labeled_batches),
                    sup_loss, self.estimate_elbo(),
                    self.timer.format_elapsed_epoch(), i / self.timer.elapsed_epoch(),
                    self.timer.format_eta(i, len(labeled_batches))))

    def train_epoch_unsup(self):

        batches = self.batchify(self.train_treebank)
        self.num_batches = len(batches)

        for i, batch in enumerate(batches, 1):
            if self.timer.elapsed() > self.max_time:
                break
            dy.renew_cg()

            loss = self.unsupervised_step(batch)

            # Optimize objective
            loss.forward()
            loss.backward()
            self.optimizer.update()

            # Store losses
            self.logger['unsup-loss'].append(loss.value())

            self.num_updates += 1

            if i % self.print_every == 0:
                self.write_tensoboard()
                loss = np.mean(self.logger['unsup-loss'][-self.print_every:])

                print('| epoch {:3d} | step {:5d}/{:5d} ({:.1%}) | loss {:8.3f} | elbo {:8.3f} | elapsed {} | {:3.2f} updates/sec | eta {}'.format(
                    self.current_epoch, i, len(batches), i / len(batches),
                    loss, self.estimate_elbo(),
                    self.timer.format_elapsed_epoch(), i / self.timer.elapsed_epoch(),
                    self.timer.format_eta(i, len(batches))))

    def supervised_step(self, batch):
        losses = []
        for tree in batch:
            if self.lmbda == 0:
                loss = self.joint_model.forward(tree)
            else:
                post_tree = tree.cnf() if self.posterior_type == 'crf' else tree
                loss = self.joint_model.forward(tree) + self.lmbda * self.post_model.forward(post_tree)
            losses.append(loss)
        loss = dy.esum(losses) / self.batch_size
        return loss

    def unsupervised_step(self, batch):
        if self.posterior_type == 'disc':
            return self.unsupervised_step_disc(batch)
        else:
            return self.unsupervised_step_crf(batch)

    def unsupervised_step_crf(self, batch):
        loss = dy.zeros(1)
        for item in batch:
            words = item.words() if self.train_objective == 'unsup' else item

            parse, parse_logprob, samples, post_entropy = self.post_model.parse_sample_entropy(
                words, self.num_samples)
            post_logprobs = [-nll for _, nll in samples]
            joint_logprobs = self.forward_joint_model(samples)

            # combined forward computation, hopefully helping autobatching
            dy.forward(post_logprobs + joint_logprobs + [post_entropy])

            a = self.optimal_baseline_scale()
            signals = [blockgrad(logprob) for logprob in joint_logprobs]
            if self.use_argmax_baseline:
                baseline = blockgrad(-self.joint_model.forward(parse))
                centered_signals = [signal - a * baseline
                    for signal in signals]
            else:
                baselines = [np.mean(signals[:i] + signals[i+1:])
                    for i in range(len(signals))]
                centered_signals = [signal - a * baseline
                    for signal, baseline in zip(signals, baselines)]
                baseline = np.mean(baselines)  # only for logging

            if self.anneal_entropy:
                anneal = self.anneal()
            else:
                anneal = 1.

            post_loss = -(anneal * post_entropy + 1 / self.num_samples * dy.esum(
                [signal * logprob for signal, logprob in zip(centered_signals, post_logprobs)]))
            joint_loss = -dy.esum(joint_logprobs) / self.batch_size
            loss += post_loss + joint_loss

            self.logger['signal'].append(np.mean(signals))
            self.logger['baseline'].append(baseline)
            self.logger['centered'].append(np.mean(centered_signals))
            self.logger['scale'].append(a)
            self.logger['entropy'].append(post_entropy.value())
            self.logger['joint'].append(np.mean(signals))
            self.logger['post'].append(np.mean([logprob.value() for logprob in post_logprobs]))
            self.logger['post-viterbi'].append(parse_logprob.value())
            self.logger['anneal-entropy'].append(anneal)
            self.logger['unique-samples'].append(len(set([tree.linearize(False) for tree, _ in samples])))

            # print(post_entropy.value())
            # print(-np.mean([logprob.value() for logprob in post_logprobs]))
            # print('$', round(-parse_logprob.value(), 3), parse.linearize(False))
            # for tree, nll in samples:
            #     print('>', round(nll.value(), 3), tree.linearize(False))
            # print()

        return loss

    def unsupervised_step_disc(self, batch):
        loss = dy.zeros(1)
        for item in batch:
            words = item.words() if self.train_objective == 'unsup' else item

            samples = [self.post_model.sample(words, self.alpha) for _ in range(self.num_samples)]
            post_logprobs = [-nll for _, nll in samples]
            joint_logprobs = self.forward_joint_model(samples)

            # combined forward computation, hopefully helping autobatching
            dy.forward(post_logprobs + joint_logprobs)

            a = self.optimal_baseline_scale()
            signals = [blockgrad(joint_logprob - post_logprob)
                for joint_logprob, post_logprob in zip(joint_logprobs, post_logprobs)]
            if self.use_argmax_baseline:
                parse, post_nll = self.post_model.parse(words)
                baseline = blockgrad(-self.joint_model.forward(parse) + post_nll)
                centered_signals = [signal - a * baseline
                    for signal in signals]
            else:
                baselines = [np.mean(signals[:i] + signals[i+1:])
                    for i in range(len(signals))]
                centered_signals = [signal - a * baseline
                    for signal, baseline in zip(signals, baselines)]
                baseline = np.mean(baselines)  # only for logging

            post_loss = 1 / self.num_samples * dy.esum(
                [signal * logprob for signal, logprob in zip(centered_signals, post_logprobs)])
            joint_loss = -dy.esum(joint_logprobs) / self.batch_size
            loss += post_loss + joint_loss

            self.logger['signal'].append(np.mean(signals))
            self.logger['baseline'].append(baseline)
            self.logger['centered'].append(np.mean(centered_signals))
            self.logger['scale'].append(a)
            self.logger['joint'].append(np.mean(signals))
            self.logger['post'].append(np.mean([logprob.value() for logprob in post_logprobs]))

            # print(-np.mean([logprob.value() for logprob in post_logprobs]))
            # for tree, nll in samples:
            #     print('>', round(nll.value(), 3), tree.linearize(False))
            # print()

        return loss

    def fiter_duplicates(self, samples):
        counted = Counter(
            [tree.linearize() for tree, _ in samples]).most_common()
        return [(fromstring(tree), count) for tree, count in counted]

    def forward_joint_model(self, samples):
        """Compute the joint log-probabilities of the samples."""

        # avoid unnecessary computation by scoring only unique trees
        filtered_joint_logprobs = {tree.linearize(): -self.joint_model.forward(tree)
            for tree, _ in self.fiter_duplicates(samples)}

        # return logprobs in original order
        joint_logprobs = [filtered_joint_logprobs[tree.linearize()]
            for tree, _ in samples]

        return joint_logprobs

    def mlp_baseline(self, words):
        """Baseline parametrized by a feedfoward network."""
        return self.baseline_model.forward(words, self.post_model)

    def optimal_baseline_scale(self, n=100):
        """Estimate optimal scaling for baseline."""
        if self.num_updates < 20:
            # cannot yet estimate variance well
            return 1.
        else:
            # static baseline: optimal scaling exists
            baseline_values = np.array(self.logger['baseline'][-n:])
            signal_values = np.array(self.logger['signal'][-n:])
            _, cov, _, var = np.cov([signal_values, baseline_values]).ravel()
            return cov/var

    def normalize(self, signal, n=100):
        """Normalize the centered learning-signal."""
        signal_mean = np.mean(self.centered_learning_signals[-n:]) if self.num_updates > 0 else 0.
        signal_var = np.var(self.centered_learning_signals[-n:]) if self.num_updates > 1 else 1.
        signal = (signal - signal_mean) / np.sqrt(signal_var)
        return signal

    def clip(self, signal):
        if self.clip_learning_signal is not None:
            if signal.value() < self.clip_learning_signal:
                signal = dy.scalarInput(self.clip_learning_signal)
        return signal

    def estimate_elbo(self, n=100):
        """Estimate the ELBO using the past learning signals."""
        if self.posterior_type == 'crf':
            return np.mean(self.logger['signal'][-n:]) + np.mean(self.logger['entropy'][-n:])
        else:
            return np.mean(self.logger['signal'][-n:])

    def write_tensoboard(self, n=100):
        if self.train_objective == 'semisup':
            self.tensorboard_writer.add_scalar(
                'loss/total',
                np.mean(self.logger['loss'][-self.print_every:]), self.num_updates)
            self.tensorboard_writer.add_scalar(
                'loss/supervised',
                np.mean(self.logger['sup-loss'][-self.print_every:]), self.num_updates)
            self.tensorboard_writer.add_scalar(
                'loss/unsupervised-surrogate',
                np.mean(self.logger['unsup-loss'][-self.print_every:]), self.num_updates)
            self.tensorboard_writer.add_scalar(
                'loss/unsupervised-elbo', self.estimate_elbo(), self.num_updates)
        else:
            self.tensorboard_writer.add_scalar(
                'loss/unsupervised-surrogate',
                np.mean(self.logger['unsup-loss'][-self.print_every:]), self.num_updates)
            self.tensorboard_writer.add_scalar(
                'loss/unsupervised-elbo', self.estimate_elbo(), self.num_updates)

        # Write batch means as scalar
        self.tensorboard_writer.add_scalar(
            'unsup/learning-signal',
            np.mean(self.logger['signal'][-self.print_every:]),
            self.num_updates)
        self.tensorboard_writer.add_scalar(
            'unsup/baseline',
            np.mean(self.logger['baseline'][-self.print_every:]),
            self.num_updates)
        self.tensorboard_writer.add_scalar(
            'unsup/scale',
            np.mean(self.logger['scale'][-self.print_every:]),
            self.num_updates)
        self.tensorboard_writer.add_scalar(
            'unsup/centered-learning-signal',
            np.mean(self.logger['centered'][-self.print_every:]),
            self.num_updates)
        self.tensorboard_writer.add_scalar(
            'unsup/posterior-logprob',
            np.mean(self.logger['post'][-self.print_every:]),
            self.num_updates)
        self.tensorboard_writer.add_scalar(
            'unsup/joint-logprob',
            np.mean(self.logger['joint'][-self.print_every:]),
            self.num_updates)
        if self.posterior_type == 'crf':
            self.tensorboard_writer.add_scalar(
                'unsup/posterior-entropy',
                np.mean(self.logger['entropy'][-self.print_every:]),
                self.num_updates)
            self.tensorboard_writer.add_scalar(
                'unsup/posterior-viterbi',
                np.mean(self.logger['post-viterbi'][-self.print_every:]),
                self.num_updates)
            self.tensorboard_writer.add_scalar(
                'unsup/unique-samples',
                np.mean(self.logger['unique-samples'][-self.print_every:]),
                self.num_updates)
            self.tensorboard_writer.add_scalar(
                'unsup/anneal-entropy',
                np.mean(self.logger['anneal-entropy'][-self.print_every:]),
                self.num_updates)

        # Write signal statistics
        if self.num_updates > 10:
            cov, corr = self.baseline_signal_covariance()
            self.tensorboard_writer.add_scalar(
                'unsup/signal-mean',
                np.mean(self.logger['signal'][-n:]),
                self.num_updates)
            self.tensorboard_writer.add_scalar(
                'unsup/signal-variance',
                np.var(self.logger['signal'][-n:]),
                self.num_updates)
            self.tensorboard_writer.add_scalar(
                'unsup/centered-signal-mean',
                np.mean(self.logger['centered'][-n:]),
                self.num_updates)
            self.tensorboard_writer.add_scalar(
                'unsup/centered-signal-variance',
                np.var(self.logger['centered'][-n:]),
                self.num_updates)
            self.tensorboard_writer.add_scalar(
                'unsup/baseline-mean',
                np.mean(self.logger['baseline'][-n:]),
                self.num_updates)
            self.tensorboard_writer.add_scalar(
                'unsup/baseline-variance',
                np.var(self.logger['baseline'][-n:]),
                self.num_updates)
            self.tensorboard_writer.add_scalar(
                'unsup/signal-baseline-cov',
                cov,
                self.num_updates)
            self.tensorboard_writer.add_scalar(
                'unsup/signal-baseline-cor',
                corr,
                self.num_updates)
            self.tensorboard_writer.add_scalar(
                'unsup/signal-baseline-var-frac',
                1 - corr**2,
                self.num_updates)

    def baseline_signal_covariance(self, n=100):
        baseline_values = np.array(self.logger['baseline'][-n:])
        signal_values = np.array(self.logger['signal'][-n:])

        signal_var, cov, _, baseline_var = np.cov(
            [signal_values, baseline_values]).ravel()

        corr = cov / np.sqrt(baseline_var * signal_var)
        return cov, corr

    def predict_post(self, examples):
        self.post_model.eval()
        trees = []
        for gold in tqdm(examples):
            dy.renew_cg()
            tree, *rest = self.post_model.parse(gold.words())
            trees.append(tree.linearize())
        self.post_model.train()
        return trees

    def check_dev(self):
        print('Evaluating posterior F1 on development set...')
        trees = self.predict_post(self.dev_treebank)

        with open(self.dev_post_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)

        dev_post_fscore = evalb(
            self.evalb_dir, self.dev_post_pred_path, self.dev_path, self.dev_post_result_path,
            param_file=self.evalb_param_file)

        print('Evaluating joint F1 and perplexity on development set...')
        decoder = GenerativeDecoder(
            model=self.joint_model, proposal=self.post_model, num_samples=self.num_dev_samples)

        print('Sampling proposals with posterior model...')
        dev_sentences = [tree.words() for tree in self.dev_treebank]
        decoder.generate_proposal_samples(
            sentences=dev_sentences, outpath=self.dev_proposals_path)

        print('Scoring proposals with joint model...')
        trees, dev_perplexity = decoder.predict_from_proposal_samples(
            inpath=self.dev_proposals_path)

        with open(self.dev_joint_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)

        dev_joint_fscore = evalb(
            self.evalb_dir, self.dev_joint_pred_path, self.dev_path, self.dev_joint_result_path,
            param_file=self.evalb_param_file)

        self.tensorboard_writer.add_scalar(
            'dev/joint-f-score', dev_joint_fscore, self.num_updates)
        self.tensorboard_writer.add_scalar(
            'dev/post-f-score', dev_post_fscore, self.num_updates)
        self.tensorboard_writer.add_scalar(
            'dev/perplexity', dev_perplexity, self.num_updates)

        self.dev_joint_fscore = dev_joint_fscore
        self.dev_post_fscore = dev_post_fscore
        self.dev_perplexity = dev_perplexity

    def check_test(self):
        print('Evaluating posterior F1 on test set...')
        trees = self.predict_post(self.test_treebank)

        with open(self.test_post_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)

        test_post_fscore = evalb(
            self.evalb_dir, self.test_post_pred_path, self.test_path, self.test_post_result_path,
            param_file=self.evalb_param_file)

        print('Evaluating joint F1 and perplexity on test set...')
        decoder = GenerativeDecoder(
            model=self.joint_model, proposal=self.post_model, num_samples=self.num_test_samples)

        print('Sampling proposals with posterior model...')
        test_sentences = [tree.words() for tree in self.test_treebank]
        decoder.generate_proposal_samples(
            sentences=test_sentences, outpath=self.test_proposals_path)

        print('Scoring proposals with joint model...')
        trees, test_perplexity = decoder.predict_from_proposal_samples(
            inpath=self.test_proposals_path)

        with open(self.test_joint_pred_path, 'w') as f:
            print('\n'.join(trees), file=f)

        test_joint_fscore = evalb(
            self.evalb_dir, self.test_joint_pred_path, self.test_path, self.test_joint_result_path,
            param_file=self.evalb_param_file)

        self.tensorboard_writer.add_scalar(
            'test/joint-f-score', test_joint_fscore)
        self.tensorboard_writer.add_scalar(
            'test/post-f-score', test_post_fscore)
        self.tensorboard_writer.add_scalar(
            'test/perplexity', test_perplexity)

        self.test_joint_fscore = test_joint_fscore
        self.test_post_fscore = test_post_fscore
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

                'current-dev-fscore': self.dev_joint_fscore,
                'best-dev-fscore': self.dev_joint_fscore,
                'test-fscore': self.test_joint_fscore,

                'current-dev-perplexity': self.dev_perplexity,
                'best-dev-perplexity': self.dev_perplexity,
                'test-perplexity': self.test_perplexity,
            }
            json.dump(state, f, indent=4)

        with open(self.post_state_checkpoint_path, 'w') as f:
            state = {
                'model': self.posterior_type,

                'num-epochs': self.current_epoch,
                'num-updates': self.num_updates,
                'elapsed': self.timer.format_elapsed(),

                'current-dev-fscore': self.dev_post_fscore,
                'best-dev-fscore': self.dev_post_fscore,
                'test-fscore': self.test_post_fscore,

                'current-dev-perplexity': None,
                'best-dev-perplexity': None,
                'test-perplexity': None,
            }
            json.dump(state, f, indent=4)

    def finalize_model_folder(self):
        move_to_final_folder(
            self.subdir, self.model_path_base, self.dev_perplexity)

    # def unsupervised_step_disc(self, batch):
    #
    #     def sample(words):
    #         samples = [self.post_model.sample(words, self.alpha)
    #             for _ in range(self.num_samples)]
    #         trees, nlls = zip(*samples)
    #         logprob = -dy.esum(list(nlls)) / len(nlls)
    #         return trees, logprob
    #
    #     def argmax_baseline(words):
    #         """Parameter-free baseline based on argmax decoding."""
    #         tree, post_nll = self.post_model.parse(words)
    #         post_logprob = -post_nll
    #         joint_logprob = -self.joint_model.forward(tree)
    #         return dy.scalarInput(blockgrad(joint_logprob - post_logprob))
    #
    #     def baselines(words):
    #         b = dy.scalarInput(0)
    #         if self.use_mlp_baseline:
    #             b += self.mlp_baseline(words)
    #         if self.use_argmax_baseline:
    #             b += argmax_baseline(words)
    #         return b
    #
    #     # Keep track of losses
    #     losses = []
    #     baseline_losses = []
    #
    #     for words in batch:
    #         # Get samples with their \mean_y [log q(y|x)] for samples y
    #         trees, post_logprob = sample(words)
    #
    #         # Compute mean \mean_y [log p(x,y)] for samples y
    #         joint_logprob = self.joint(trees)
    #
    #         # Compute \mean_y [log p(x,y) - log q(y|x)] and detach
    #         learning_signal = blockgrad(joint_logprob - post_logprob)
    #
    #         # Substract baseline
    #         baseline = baselines(words)
    #         a = self.optimal_baseline_scale()
    #         centered_learning_signal = learning_signal - a * baseline
    #
    #         # Normalize
    #         normalized_learning_signal = self.normalize(centered_learning_signal)
    #
    #         # Optional clipping of learning signal
    #         normalized_learning_signal = self.clip(normalized_learning_signal)
    #
    #         baseline_loss = centered_learning_signal**2
    #         post_loss = -blockgrad(normalized_learning_signal) * post_logprob
    #         joint_loss = -joint_logprob
    #         loss = post_loss + joint_loss
    #
    #         losses.append(loss)
    #         baseline_losses.append(baseline_loss)
    #
    #         # For tesorboard logging
    #         self.logger['signal'].append(learning_signal)
    #         self.logger['baseline'].append(baseline.value())
    #         self.logger['scale'].append(a)
    #         self.logger['centered'].append(centered_learning_signal.value())
    #         self.logger['normalized'].append(normalized_learning_signal.value())
    #         self.logger['post'].append(post_logprob.value())
    #         self.logger['joint'].append(joint_logprob.value())
    #
    #     # save the batch average
    #     self.learning_signals.append(
    #         np.mean(self.logger['signal'][-self.batch_size:]))
    #     self.baseline_values.append(
    #         np.mean(self.logger['baseline'][-self.batch_size:]))
    #     self.centered_learning_signals.append(
    #         np.mean(self.logger['centered'][-self.batch_size:]))
    #
    #     # Average losses over minibatch
    #     loss = dy.esum(losses) / self.batch_size
    #     baseline_loss = dy.esum(baseline_losses) / self.batch_size
    #
    #     return loss, baseline_loss

    # def unsupervised_step_crf(self, batch):
    #
    #     def argmax_baseline(tree):
    #         """Parameter-free baseline based on argmax decoding."""
    #         joint_logprob = -self.joint_model.forward(tree)
    #         return dy.scalarInput(blockgrad(joint_logprob))
    #
    #     def baselines(words, tree):
    #         b = dy.scalarInput(0)
    #         if self.use_mlp_baseline:
    #             b += self.mlp_baseline(words)
    #         if self.use_argmax_baseline:
    #             b += argmax_baseline(tree)
    #         return b
    #
    #     # Keep track of losses
    #     losses = []
    #     baseline_losses = []
    #
    #     for item in batch:
    #
    #         # TODO: an ugly hack to deal with the inconsintency between unsup and semisup
    #         if self.train_objective == 'unsup':
    #             words = item.words()
    #         else:
    #             words = item
    #
    #         if self.exact_entropy:
    #             # Combined computation is most efficient
    #             parse, samples, post_entropy = self.post_model.parse_sample_entropy(
    #                 words, self.num_samples)
    #         else:
    #             parse, samples = self.post_model.parse_sample(words, self.num_samples)
    #
    #         # Compute \mean_y [log p(y|x)] for samples
    #         trees, post_nlls = zip(*samples)
    #         post_logprob = - dy.esum(list(post_nlls)) / len(post_nlls)
    #
    #         for tree in trees:
    #             print(tree.linearize(with_tag=False))
    #
    #         # Compute mean \mean_y [log p(x,y)] for samples y
    #         joint_logprob = self.joint(trees)
    #
    #         # Compute \mean_y [log p(x,y)]
    #         if self.exact_entropy:
    #             learning_signal = blockgrad(joint_logprob)
    #         else:
    #             learning_signal = blockgrad(joint_logprob - post_logprob)
    #
    #         # Substract baseline
    #         baseline = baselines(words, parse)
    #         a = self.optimal_baseline_scale()
    #         centered_learning_signal = learning_signal - a * baseline
    #
    #         # Normalize
    #         normalized_learning_signal = self.normalize(centered_learning_signal)
    #
    #         # Optional clipping of learning signal
    #         normalized_learning_signal = self.clip(normalized_learning_signal)
    #
    #         if self.exact_entropy:
    #             post_loss = -(blockgrad(normalized_learning_signal) * post_logprob + post_entropy)
    #         else:
    #             post_loss = -blockgrad(normalized_learning_signal) * post_logprob
    #         joint_loss = -joint_logprob
    #         loss = post_loss + joint_loss
    #         baseline_loss = centered_learning_signal**2
    #
    #         losses.append(loss)
    #         baseline_losses.append(baseline_loss)
    #
    #         # For tesorboard logging
    #         self.logger['signal'].append(learning_signal)
    #         self.logger['baseline'].append(baseline.value())
    #         self.logger['scale'].append(a)
    #         self.logger['centered'].append(centered_learning_signal.value())
    #         self.logger['normalized'].append(normalized_learning_signal.value())
    #         self.logger['post'].append(post_logprob.value())
    #         self.logger['joint'].append(joint_logprob.value())
    #         if self.exact_entropy:
    #             self.logger['entropy'].append(post_entropy.value())
    #
    #     self.learning_signals.append(
    #         np.mean(self.logger['signal'][-self.batch_size:]))
    #     self.baseline_values.append(
    #         np.mean(self.logger['baseline'][-self.batch_size:]))
    #     self.centered_learning_signals.append(
    #         np.mean(self.logger['centered'][-self.batch_size:]))
    #
    #     # Average losses over minibatch
    #     loss = dy.esum(losses) / self.batch_size
    #     baseline_loss = dy.esum(baseline_losses) / self.batch_size
    #
    #     return loss, baseline_loss

    # def unsupervised_step_crf(self, batch):
    #     post_logprobs = []
    #     joint_logprobs = []
    #     post_entropies = []
    #     baselines = []
    #     for item in batch:
    #
    #         words = item.words() if self.train_objective == 'unsup' else item
    #
    #         parse, samples, post_entropy = self.post_model.parse_sample_entropy(
    #             words, self.num_samples)
    #         post_logprob = -dy.esum(
    #             [nll for _, nll in samples]) / self.num_samples
    #         joint_logprob = -dy.esum(
    #             [self.joint_model.forward(tree) for tree, _ in samples]) / self.num_samples
    #         argmax_logprob = -self.joint_model.forward(parse)
    #
    #         post_logprobs.append(post_logprob)
    #         joint_logprobs.append(joint_logprob)
    #         post_entropies.append(post_entropy)
    #         baselines.append(argmax_logprob)
    #
    #         # print()
    #         # for tree, _ in samples:
    #         #     print('>', tree.linearize(False))
    #
    #     # forward all computations at once, hopefully helping autobatch
    #     dy.forward(post_logprobs + joint_logprobs + post_entropies + baselines)
    #
    #     a = self.optimal_baseline_scale()
    #     signals = [blockgrad(logprob) for logprob in joint_logprobs]
    #     baselines = [blockgrad(logprob) for logprob in baselines]
    #     centered_signals = [signal - a * baseline
    #         for signal, baseline in zip(signals, baselines)]
    #
    #     if self.normalize_learning_signal:
    #         centered_signals = self.normalize(centered_signals)
    #
    #     post_loss = -dy.esum([signal * post_logprob + post_entropy for
    #         signal, post_logprob, post_entropy in zip(centered_signals, post_logprobs, post_entropies)]) / self.batch_size
    #     joint_loss = -dy.esum(joint_logprobs) / self.batch_size
    #     baseline_loss = np.mean(centered_signals)**2
    #     loss = post_loss + joint_loss
    #
    #     self.logger['signal'].append(np.mean(signals))
    #     self.logger['baseline'].append(np.mean(baselines))
    #     self.logger['centered'].append(np.mean(centered_signals))
    #     self.logger['scale'].append(a)
    #     self.logger['entropy'].append(np.mean([entropy.value() for entropy in post_entropies]))
    #     self.logger['post'].append(np.mean([logprob.value() for logprob in post_logprobs]))
    #     self.logger['joint'].append(np.mean([logprob.value() for logprob in joint_logprobs]))
    #
    #     return loss

    # def unsupervised_step_disc(self, batch):
    #     post_logprobs = []
    #     joint_logprobs = []
    #     baselines = []
    #     for item in batch:
    #
    #         words = item.words() if self.train_objective == 'unsup' else item
    #
    #         samples = [self.post_model.sample(words, self.alpha) for _ in range(self.num_samples)]
    #
    #         post_logprob = -dy.esum(
    #             [nll for _, nll in samples]) / self.num_samples
    #         joint_logprob = -dy.esum(
    #             [self.joint_model.forward(tree) for tree, _ in samples]) / self.num_samples
    #
    #         parse, post_nll = self.post_model.parse(words)
    #         argmax_logprob = -self.joint_model.forward(parse) + post_nll
    #
    #         post_logprobs.append(post_logprob)
    #         joint_logprobs.append(joint_logprob)
    #         baselines.append(argmax_logprob)
    #
    #     # forward all computations at once, hopefully helping autobatch
    #     dy.forward(post_logprobs + joint_logprobs + baselines)
    #
    #     a = self.optimal_baseline_scale()
    #     signals = [blockgrad(joint_logprob - post_logprob)
    #         for joint_logprob, post_logprob in zip(joint_logprobs, post_logprobs)]
    #     baselines = [blockgrad(baseline) for baseline in baselines]
    #     centered_signals = [signal - a * baseline
    #         for signal, baseline in zip(signals, baselines)]
    #
    #     if self.normalize_learning_signal:
    #         centered_signals = self.normalize(centered_signals)
    #
    #     post_loss = -dy.esum([signal * post_logprob for
    #         signal, post_logprob in zip(centered_signals, post_logprobs)]) / self.batch_size
    #     joint_loss = -dy.esum(joint_logprobs) / self.batch_size
    #     baseline_loss = np.mean(centered_signals)**2
    #     loss = post_loss + joint_loss
    #
    #     self.logger['signal'].append(np.mean(signals))
    #     self.logger['baseline'].append(np.mean(baselines))
    #     self.logger['centered'].append(np.mean(centered_signals))
    #     self.logger['scale'].append(a)
    #     self.logger['post'].append(np.mean([logprob.value() for logprob in post_logprobs]))
    #     self.logger['joint'].append(np.mean([logprob.value() for logprob in joint_logprobs]))
    #
    #     return loss
