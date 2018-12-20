"""
Softmax approximation with Complementary Sum Sampling.
Follows Botev et al. 2017
    http://web4.cs.ucl.ac.uk/staff/D.Barber/publications/AISTATS2017.pdf.
"""
import dynet as dy
import numpy as np


def css(logits, target, vocab, num_samples=250):
    """
    Computes the negative log-likelihood with a CSS approximate of the Softmax.
    Args:
        logits(dynet.Expression): the output logits
        target(int): the index of the target
    Returns:
        dynet.Expression: negative log-likelihood.
    """
    # Obtain the positive and negative set.
    neg_dim = vocab.size - 1
    probs = np.ones(vocab.size) / neg_dim
    probs[target] = 0
    negative_set = np.random.choice(
        vocab.size, num_samples, replace=False, p=probs).tolist()
    positive_set = [target]

    log_kappa = np.log(neg_dim) - np.log(num_samples)

    # normalizer = dy.pick_batch(logits, positive_set + negative_set)

    # Compute the approximate lognormalizer.
    # a = dy.max_dim(normalizer)
    # lognormalizer = a + dy.log(
    #     num_samples * log_kappa + dy.sum_batches(dy.exp(normalizer - a)))
    # lognormalizer = dy.logsumexp(normalizer)

    # TODO: This is horrible...
    lognormalizer = dy.logsumexp([log_kappa + logits[0], log_kappa + logits[1]])
    for i in negative_set[2:]:
        lognormalizer = dy.logsumexp([lognormalizer, log_kappa + logits[i]])
    lognormalizer = dy.logsumexp([lognormalizer, logits[target]])

    # We return the negative log likelihood
    logprob = logits[target] - lognormalizer
    return -logprob
