# Thesis

A stochastic decoder for generative parsing with Recurrent Neural Network Grammars.

Or: a neural language model by marginalizing parse trees.

## Project description

The recently proposed Recurrent Neural Network Grammar (RNNG) is a top-down transition-based parser parametrized using Recurrent Neural Networks. There is a discriminative and generative variant, and can parse both constituency and dependency grammars. The generative variant can additionally be used for language modeling, in which case the trained discriminative parser is used as proposal distribution over trees in an importance sampling scheme that is used to evaluate the marginal probability of the words. The trees are sampled using ancestral sampling over sequence of transition actions.

The stochastic decoder is a latent variable neural language model recently proposed in the context of neural machine translation (NMT). The model is an conditional language model augmented with latent gaussian variables on the target side to model explicitly model variation in the training data. Posterior inference is performed with amortized variational inference in the same line as the variational auto-encoder (VAE).

This project proposes to replace the RNN in the RNNG with the stochastic RNN with the aim of training the discriminative parser for more variability. We conjecture that explicitly modeling this variability will make the discriminative parser a better proposal distribution for the generative parser. As such, this could improve the generative parser as language model.

Additionally, the incorporation of latent variables gives us the ability to incorporate inductive bias in the priors. A first direction would be to put priors on the latent states of the stochastic decoder that induce sparsity. Replacing the Gaussian prior with a Dirichlet that has a sparsity inducing hyperparameter (inference can be performed using a log-normal approximation to the Dirichlet following [Autoencoding Variational Inference for Topic Models](https://arxiv.org/pdf/1703.01488.pdf)) can result in a posterior on the latent-states that are easier to interpret.

An ambitious extension of this project is to incorporate the learning of the posterior distribution q(y|x) during training of the generative parser.




## Logbook

I will start using a logbook in this google docs. I will share it with Wilker, and perhaps also with Chris. The logbook will help me plan and evaluate my time. Also, it will give me the structure I desire so badly at this moment.

[Here's the link to the docs](https://docs.google.com/document/d/131-qsS-20-ZAEMkRGx1XikZoUR9KdCTPKHYoG-GSrTQ/edit?usp=sharing).

I will use the logbook for the following things:

* At the beginning of each day I select a small number of things from the to-do list below to work on. I write thes in the logbook.
* At the end of each day I will write down what I did, and roughly how much time it took to do it. It will be difficult to keep myself to this! But, it will guide me, make me more focussed, and hopefully motivate me.
* I write down a summary of my meeting with Wilker.

## Project proposal

The proposal can be read in [here](doc/outline).

## Planning

The planning can also be found [here](doc/outline).

## Evaluation

The RNNG appears to get good perplexity scores. But is this the metric by which we want to evaluate the language model? Perplexity measures the average per word surprisal, and thus reflects an average case of succes. But note:
> NLP people are very happy when we do well on the average case, and linguists are concerned how well we do in the rare, long-tail phenomena. - Chris Dyer [source](https://youtu.be/hIlR7hIAzi8?t=11m12s)

Hence, we can perhaps better evaluate the model the RNNG on phenomena that are known to be hard to handle for linear RNN-based models. Examples of this are subject-verb agreement with long intervening material:
> The keys is/are on the table.
> The keys to the cabinet is/are on the table.
> The keys to the cabinet in the closet is/are on the table.
> [Linzen, Dupoux, Goldberg (2016)](https://arxiv.org/pdf/1611.01368.pdf)




## Todo

A list of things to do.

Proposal + outline:
- [ ] Setup minimal latex workspace: figure out bibtex manager, make folder structure, and minimal style-sheet.
- [ ] Write the latex outline. Combine, reorder, and expand the various markdown files that you wrote so far.
- [ ] Send the latex outline to Wilker (maybe an open sharelatex would be best? This way he can incorporate feedback live) and then to Christian.
- [ ] Keep getting back to this outline - refine it as you go along. This will be the seed of your final thesis!

Coding
- [ ] Set up a good work-situation in the `src` folder.
- [ ] Get PTB data.
- [ ] Write out a plan to
