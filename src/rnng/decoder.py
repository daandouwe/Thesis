import os
import json
from collections import Counter
from collections import defaultdict
from tqdm import tqdm

import dynet as dy
import numpy as np

from .model import DiscRNNG, GenRNNG
from crf.model import ChartParser
from utils.trees import fromstring, add_dummy_tags
from utils.general import ceil_div


class GenerativeDecoder:
    """Decoder for generative RNNG by importance sampling."""
    def __init__(
            self,
            model=None,
            proposal=None,
            num_samples=100,
            alpha=0.8,
    ):
        if model is not None:
            assert isinstance(model, GenRNNG)
            model.eval()
        if proposal is not None:
            assert (isinstance(proposal, DiscRNNG) or
                isinstance(proposal, ChartParser))
            proposal.eval()

        self.model = model
        self.proposal = proposal
        self.num_samples = num_samples
        self.alpha = alpha
        self.use_argmax = (num_samples == 1)
        self.use_loaded_samples = (self.proposal is None)

    def parse(self, words):
        """Return the estimated MAP tree for the words."""
        return self.map_tree(words)

    def map_tree(self, words):
        """Estimate the MAP tree."""
        scored = self.scored_samples(words)
        ranked = sorted(scored, reverse=True, key=lambda t: t[2])
        best_tree, proposal_logprob, joint_logprob, count = ranked[0]
        return best_tree, proposal_logprob, joint_logprob

    def logprob(self, words):
        """Estimate the log probability."""
        if self.use_argmax:
            tree, proposal_logprob, joint_logprob = self.scored_argmax(words)
            logprob = joint_logprob - proposal_logprob
        else:
            scored = self.scored_samples(words)
            weights, counts = np.zeros(len(scored)), np.zeros(len(scored))
            for i, (_, proposal_logprob, joint_logprob, count) in enumerate(scored):
                weights[i] = joint_logprob - proposal_logprob
                counts[i] = count
            a = weights.max()
            logprob = a + np.log(np.mean(np.exp(weights - a) * counts))  # log-mean-exp for stability
        return logprob

    def perplexity(self, words):
        """Estimate the perplexity."""
        return np.exp(-self.logprob(words) / len(list(words)))

    def remove_duplicates(self, samples):
        """Filter out duplicate trees from the samples."""
        output = []
        seen = set()
        for tree, logprob in samples:
            if tree.linearize() not in seen:
                output.append((tree, logprob))
                seen.add(tree.linearize())
        return output

    def count_samples(self, samples):
        """Filter out duplicate trees from the samples."""
        counts = Counter([tree.linearize() for tree, _ in samples])
        filtered = self.remove_duplicates(samples)
        return [(tree, logprob, counts[tree.linearize()]) for tree, logprob in filtered]

    def scored_argmax(self, words):
        """Score the proposal's argmax tree."""
        tree, proposal_nll = self.proposal.parse(words)
        joint_logprob = -self.model.forward(tree, is_train=False)
        proposal_logprob = -proposal_nll
        return tree, proposal_logprob.value(), joint_logprob.value()

    def scored_samples(self, words):
        """Return a list of proposal samples that will be scored by the joint model."""
        if self.use_loaded_samples:
            samples = next(self.samples)
        else:
            dy.renew_cg()
            words = list(words)
            samples = []
            for _ in range(self.num_samples):
                tree, nll = self.proposal.sample(words, alpha=self.alpha)
                samples.append((tree, -nll.value()))

        # Count and filter
        samples = self.count_samples(samples)  # list of tuples (tree, post_logprob, count)

        scored = []
        for tree, proposal_logprob, count in samples:
            dy.renew_cg()
            joint_logprob = -self.model.forward(tree, is_train=False).value()
            scored.append((tree, proposal_logprob, joint_logprob, count))

        return scored

    def read_proposals(self, path):
        print(f'Loading discriminative (proposal) samples from `{path}`...')
        with open(path) as f:
            lines = [line.strip() for line in f.readlines()]
        sent_id = 0
        samples = []
        proposals = []
        for line in lines:
            sample_id, logprob, tree = line.split('|||')
            sample_id, logprob, tree = int(sample_id), float(logprob), fromstring(add_dummy_tags(tree.strip()))
            if sample_id > sent_id:
                # Arrived at the first sample of next sentence
                assert len(samples) == self.num_samples, f'not enough samples for line {sample_id}'
                proposals.append(samples)
                sent_id = sample_id
                samples = []
            samples.append((tree, logprob))
        proposals.append(samples)
        return proposals

    def load_proposal_samples(self, path):
        """Load proposal samples that were written to file."""
        assert os.path.exists(path), path

        self.samples = iter(self.read_proposals(path))
        self.use_loaded_samples = True

    def load_proposal_model(self, dir):
        """Load the proposal model to sample with."""
        assert os.path.isdir(dir), dir

        print(f'Loading proposal model from `{dir}`...')
        model_checkpoint_path = os.path.join(dir, 'model')
        state_checkpoint_path = os.path.join(dir, 'state.json')
        [proposal] = dy.load(model_checkpoint_path, dy.ParameterCollection())

        assert isinstance(proposal, DiscRNNG), f'expected discriminative model got {type(proposal)}'

        with open(state_checkpoint_path, 'r') as f:
            state = json.load(f)
            epochs = state['epochs']
            fscore = state['test-fscore']

        print(f'Loaded model trained for {epochs} epochs with test-fscore {fscore}.')

        self.proposal = proposal
        self.proposal.eval()
        self.use_loaded_samples = False

    def generate_proposal_samples(self, examples, path):
        # TODO: make this more general: for sentences, not just trees!
        """Use the proposal model to generate proposal samples."""
        samples = []
        for i, tree in enumerate(tqdm(examples)):
            dy.renew_cg()
            for _ in range(self.num_samples):
                tree, nll = self.proposal.sample(tree.words(), alpha=self.alpha)
                samples.append(
                    ' ||| '.join((str(i), str(-nll.value()), tree.linearize(with_tag=False))))

        with open(path, 'w') as f:
            print('\n'.join(samples), file=f, end='')

    def predict_from_proposal_samples(self, path):
        """
        Predict MAP trees and perplexity from proposal samples in path.
        Especially useful for evaluation during training.
        """

        # Load scored proposal samples
        all_samples = defaultdict(list)  # i -> [samples for sentence i]
        with open(path) as f:
            for line in f:
                i, proposal_logprob, tree = line.strip().split(' ||| ')
                i, proposal_logprob, tree = int(i), float(proposal_logprob), fromstring(add_dummy_tags(tree.strip()))
                all_samples[i].append((tree, proposal_logprob))

        # Score the trees
        for i, samples in tqdm(all_samples.items()):
            # Count and remove duplicates
            samples = self.count_samples(samples)
            scored_samples = []
            for (tree, proposal_logprob, count) in samples:
                dy.renew_cg()
                joint_logprob = -self.model.forward(tree, is_train=False).value()
                scored_samples.append(
                    (tree, proposal_logprob, joint_logprob, count))
            all_samples[i] = scored_samples

        # Get the predictions
        trees = []
        nlls = []
        lengths = []
        for i, scored in all_samples.items():
            # Estimate the map tree by sorting the scored tuples according to the joint logprob
            ranked = sorted(scored, reverse=True, key=lambda t: t[2])
            tree, _, _, _ = ranked[0]

            # Estimate the perplexity
            weights, counts = np.zeros(len(scored)), np.zeros(len(scored))
            for i, (_, proposal_logprob, joint_logprob, count) in enumerate(scored):
                weights[i] = joint_logprob - proposal_logprob
                counts[i] = count
            a = weights.max()
            logprob = a + np.log(np.mean(np.exp(weights - a) * counts))  # log-mean-exp for stability

            trees.append(tree.linearize())  # the estimated MAP tree
            nlls.append(-logprob)  # the estimate for -log p(x)
            lengths.append(len(tree.words()))  # need for computing total perplexity

        # NOTE: we compute perplexity as average over words,
        # which is correct, and not averaged over sentences.
        perplexity = np.exp(np.sum(nlls) / np.sum(lengths))

        return trees, perplexity
