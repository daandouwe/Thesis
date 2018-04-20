# Thesis

A stochastic decoder for generative parsing with Recurrent Neural Network Grammars.

Or: a neural language model by marginalizing parse trees.

## Project description

The recently proposed Recurrent Neural Network Grammar (RNNG) is a top-down transition-based parser parametrized using Recurrent Neural Networks. There is a discriminative and generative variant. The generative variant can be used for language modeling. In this case the trained discriminative parser is used as proposal distribution over trees in an importance sampling scheme that is used to evaluate the marginal probability of the words. The trees are sampled using ancestral sampling over sequence of transition actions.

The stochastic decoder is a latent variable neural language model recently proposed in the context of neural machine translation (NMT). The model is an conditional language model augmented with latent gaussian variables on the target side to model explicitly model variation in the training data. Posterior inference is performed with amortized variational inference in the same line as the variational auto-encoder (VAE).

This project proposes to replace the RNN in the RNNG with the stochastic RNN with the aim of training the discriminative parser for more variability. We conjecture that explicitly modeling this variability will make the discriminative parser a better proposal distribution for the generative parser. As such, this could improve the generative parser as language model.

Additionally, the incorporation of latent variables gives us the ability to incorporate inductive bias in the priors. A first direction would be to put priors on the latent states of the stochastic decoder that induce sparsity. Replacing the Gaussian prior with a Dirichlet that has a sparsity inducing hyperparameter (inference can be performed using a log-normal approximation to the Dirichlet following [Autoencoding Variational Inference for Topic Models](https://arxiv.org/pdf/1703.01488.pdf)) can result in a posterior on the latent-states that are easier to interpret.

An ambitious extension of this project is to incorporate




## Logbook

I will start using a logbook in this google docs. I will share it with Wilker, and perhaps also with Chris. The logbook will help me plan and evaluate my time. Also, it will give me the structure I desire so badly at this moment.

I will use the logbook for the following things:

* At the beginning of each day I select a small number of things from the to-do list below to work on.
* At the end of each day I will write down what I did, and roughly how much time it took to do it. It will be difficult to keep myself to this! But, it will guide me, make me more focussed, and hopefully motivate me.
* I write down a summary of my meeting with Wilker.

## Project proposal

The proposal can be read in [here](doc/outline).

## Planning

The planning can also be found [here](doc/outline).

## Todo

A list of things to do.

Proposal + outline:
[ ] Setup minimal latex workspace: figure out bibtex manager, make folder structure, and minimal style-sheet.
[ ] Write the latex outline. Combine, reorder, and expand the various markdown files that you wrote so far.
[ ] Send the latex outline to Wilker (maybe an open sharelatex would be best? This way he can incorporate feedback live)

Coding
[ ] Set up a good work-situation in the `src` folder.
[ ] Get PTB Data
[ ] 
