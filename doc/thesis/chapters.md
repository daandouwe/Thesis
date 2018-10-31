# Chapter outline

## Chapter 1: RNNGs
- Explain model
- Describe variants
- Analysis of performance on challengin questions (Linzen at al. and Goldberg)
- Psych

## Implementation
- Challenges in implementation
- Autobatching as solution

## Factorizing joint model
We factorize the joint model p(x,y).
- Factorize p(x, y) = p(t)p(x|t)
- Plus latent variable z: p(x,y) = \int p(t)p(x|t,z) p(z) dz
- Choosing p(x|t) and p(x|t,z)
  - p(x|t) = \prod_i p(x_i|t)
  - p(x|t) = \prod_i p(x_i|t, x_{<i})
  - p(x|t,z) = \prod_i p(x_i|t, z)
  - p(x|t,z) = \prod_i p(x_i|t, z, x_{<i})
- p(x|t) is a Generative RNNG where only the `GEN` actions are modelled.

## Latent variables for composition
The composition function can be altered.
- Latent Factor composition using Concrete samples.
- Idea: SparseMAP for attention

## Semisupervised
