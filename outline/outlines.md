
# RNNG with stochastic (RNN) decoder

This is the project that Wilker proposed.

## Introduction
This research combines [Recurrent Neural Network Grammars](https://arxiv.org/abs/1602.07776) and [A Stochastic Decoder for NMT](https://arxiv.org/abs/1602.07776).

The RNNG is a parsing model that makes no Markov assumption. The RNNG uses as shift-reduce parser (stack, buffer, etc.) where the decisions are parametrized by RNNs that condition on the entire syntactic derivation history. Dyer proposed a discriminative and a generative variant. Parsing models that make no Markov assumption are a good testbed for the stochastic RNN model that Philip and I propose in our ACL submission.

The discriminative model can be used to parse, and is straightforward to train. The generative model is harder to train and uses importance sampling, but can additionally be used as a language model, if you evaluate p(x) by marginalizing over all latent trees that generate x. The RNNG can then be used as a syntactic language model to generate text. **This direction interests me the most**.

The RNN that parametrizes the parse decisions (`gen(x)` and `reduce`) can be replaced by the stochastic RNN, which is trained with VI.

## Constituency or dependency

Since the RNNG is a


## Aside

Wilker noted that

> The generative variant is very hard to train (even with variational inference). I find Dyer's strategy unsatisfactory even though seemly effective (the fact that it works also puzzles me a bit).

Can we come up with another method of training the generative parser? This could be an interesting challenge.

## Research questions

1. Can the stochastic decoder make the discriminative parser more robust for out of domain parsing?
2. Can the stochastic decoder let the generative parser create more variable sentences?
3. Can we come up with another way of training the generative parser?
4. Can we use the (stochastic) generative parser as a decoder (conditional language model) for NMT? E.g. [Learning to Parse ]


## Outline

1. Replicate the original paper. Focus on the discriminative variant in the beginning, and then continue to the generative model.
2. Replace the RNNs with SRNNs to create the S-RNNG.
3. Use the generative S-RNNG as a decoder for neural NMT.

## Possible directions








#
