# Mini-projects

## LDA++
* `Done` LDA with collapsed Gibss sampler
* `In progress` LDA with VI
* `In progress` LDA with stochastic VI
* `In progress` LDA with VAE
* Dynamic LDA with VI
* Correlated LDA with VI
* Deep exponential family / Pachinko allocation with Black Box VI

## sLDS
* Structured VAE


##


# Models / Applications

## Topic models
The topic model zoo.
* `Done` A **review** by David Blei: [Probabilistic Topic Models (2012)](http://delivery.acm.org/10.1145/2140000/2133826/p77-blei.pdf?ip=89.99.242.224&id=2133826&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1518968830_1cea7e1bfa5e86d1d3e1cebdcde81c2b).
* A **course** on topic models: [Advanced Topic Modeling](https://mimno.infosci.cornell.edu/info6150/)
* A **thesis** dedicated VAE type methods for LDA (supervised by Max Welling): [SGVB Topic Modeling](https://esc.fnwi.uva.nl/thesis/centraal/files/f1573659295.pdf)

### LDA
* `Done` The model with mean field VI: [Latent Dirichlet Allocation (2003)](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)
* `Done` With Collapsed Gibbs sampling: [Finding scientific topics](http://www.pnas.org/content/pnas/101/suppl_1/5228.full.pdf)
* `Done` With stochastic mean field VI: [Online Learning for Latent Dirichlet Allocation](https://papers.nips.cc/paper/3902-online-learning-for-latent-dirichlet-allocation.pdf)

### Correlated LDA
* [Correlated Topic Models](http://www.cs.columbia.edu/~blei/papers/BleiLafferty2006.pdf)
* [A correlated topic model of Science](http://www.cs.columbia.edu/~blei/papers/BleiLafferty2007.pdf)

### Dynamic LDA
* [Dynamic Topic Models (2006)](http://www.cs.columbia.edu/~blei/papers/BleiLafferty2006a.pdf)
* **Video** [David Blei lecture](https://www.youtube.com/watch?v=7BMsuyBPx90)

### Syntactic LDA
* [Integrating Topics and Syntax (2005)](https://papers.nips.cc/paper/2587-integrating-topics-and-syntax.pdf)

### Pachinko Allocation
* [Pachinko Allocation (2006)](https://people.cs.umass.edu/~mccallum/papers/pam-icml06.pdf)

## Word embeddings
The word embedding zoo.
* An **overview**: [Levy & Goldberg (2015)](http://www.aclweb.org/anthology/Q15-1016)
* Another **overview**: [New Directions in Vector Space Models of Meaning](http://www.cs.ox.ac.uk/files/6605/aclVectorTutorial.pdf)

### SGNS
* `Done` [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

### GloVe
* `Done` [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)

### PPMI-SVD
* `Done` [Improving Distributional Similarity with Lessons Learned from Word Embeddings](http://www.aclweb.org/anthology/Q15-1016)

### Gaussian embeddings
* [Word Representations via Gaussian Embedding](https://arxiv.org/pdf/1412.6623.pdf)

### Stick-breaking (adaptive) skip-gram
* [Breaking Sticks and Ambiguities with Adaptive Skip-gram](https://arxiv.org/pdf/1502.07257.pdf)

### EmbedAllign
* Deep Generative Model for Joint Alignment and Word Representation

## Topic models++
### +Gaussians
* [Gaussian LDA for Topic Models with Word Embeddings (2015)](http://www.aclweb.org/anthology/P15-1077)

### +dense representaions
* [Improving Topic Models with Latent Feature Word Representations](http://www.aclweb.org/anthology/Q15-1022)
* [Mixing Dirichlet Topic Models and Word Embeddings to Make lda2vec](https://arxiv.org/pdf/1605.02019.pdf)
* [Blog-post](https://multithreaded.stitchfix.com/blog/2016/05/27/lda2vec/#topic=38&lambda=1&term=)

### +RNN/LSTM (!)
* [Latent LSTM allocation](http://proceedings.mlr.press/v70/zaheer17a/zaheer17a.pdf)
* [TopicRNN: A recurrent neural network with long-range semantic dependency](http://www.columbia.edu/~jwp2128/Papers/DiengWangetal2017.pdf)

### +VAE inference
* [Discovering Discrete Latent Topics with Neural Variational Inference](https://arxiv.org/pdf/1706.00359.pdf)
* [Autoencoding Variational Inference For Topic Models](https://arxiv.org/pdf/1703.01488.pdf)

## Style transfer
* [Style Transfer from Non-Parallel Text by Cross-Alignment](https://arxiv.org/pdf/1705.09655.pdf)

## Controlled text generation
* [Generating Sentences from a Continuous Space](https://arxiv.org/pdf/1511.06349.pdf)
* [Language as a Latent Variable: Discrete Generative Models for Sentence Compression](https://arxiv.org/pdf/1609.07317.pdf)

## Some nonparametric 'language models'
* [A Bayesian Framework for Word Segmentation: Exploring the Effects of Context](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.5361&rep=rep1&type=pdf)
* [Producing Power-Law Distributions and Damping Word Frequencies with Two-Stage Language Models (2011)](http://www.jmlr.org/papers/volume12/goldwater11a/goldwater11a.pdf)

## Non-language models
* Latent switching linear dynamical systems (SLDS), e.g. segmenting video
    * [A Linear Dynamical System Model for Text](https://arxiv.org/pdf/1502.04081.pdf)
    * [Recurrent Switching Linear Dynamical Systems](https://arxiv.org/pdf/1603.00788.pdf)
* Mixture density networks (e.g. modelling handwriting at [Otoro](http://blog.otoro.net/2015/12/12/handwriting-generation-demo-in-tensorflow/) and [Distill](https://distill.pub/2016/handwriting/)).

# Graphical models refresher
* [Build, Compute, Critique, Repeat: Data Analysis with Latent Variable Models (Blei 2014)](http://www.cs.columbia.edu/~blei/papers/Blei2014b.pdf)


# Variational inference (VI)
Overview:
* `Done` **Review** [Variational Inference: A Review for Statisticians (2017)](https://arxiv.org/pdf/1601.00670.pdf)
* `Done` **Video** [Variational Inference: Foundations and Modern Methods](https://www.youtube.com/watch?v=dGVEtq34jTU)
* `Done` **Blog** [How does physics connect to machine learning?](https://jaan.io/how-does-physics-connect-machine-learning/)

## Mean field coordinate ascent VI
* `Done` [Variational Inference: A Review for Statisticians (2017)](https://arxiv.org/pdf/1601.00670.pdf)
* [Build, Compute, Critique, Repeat: Data Analysis with Latent Variable Models (section 4)](http://www.cs.columbia.edu/~blei/papers/Blei2014b.pdf)
* [Beal's thesis chapter 2: Variational Bayesian Theory](https://www.cse.buffalo.edu//faculty/mbeal/thesis/beal03_2.pdf)

## Stochastic VI
* `Done` [Stochastic Variational Inference (2013)](https://arxiv.org/pdf/1206.7051.pdf)
* `Done` [Online Learning for Latent Dirichlet Allocation (2010)](https://papers.nips.cc/paper/3902-online-learning-for-latent-dirichlet-allocation.pdf)

## Amortized VI
### VAE
* `Done` [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)
* [Stochastic Backpropagation and Approximate Inference in Deep Generative Models](https://arxiv.org/pdf/1401.4082.pdf)
* Kingma's thesis [chapter 2](https://pure.uva.nl/ws/files/17891313/Thesis.pdf)
* `Done` **Video**: [Kingma talk](https://www.youtube.com/watch?v=rjZL7aguLAs)
* `Done` **Tutorial** [Tutorial - What is a VAE?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)
* **Blog** Otoro: [Generating Large Images from Latent Vectors Part 1](http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/) and [Part 2](http://blog.otoro.net/2016/06/02/generating-large-images-from-latent-vectors-part-two/).

### Discrete VAE
* [Discrete Variational Autoencoders](https://arxiv.org/pdf/1609.02200.pdf)
* [Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937.pdf)

### Stick-breaking VAE
* `Done` [Stick-Breaking Variational Autoencoders](https://arxiv.org/pdf/1605.06197.pdf)

### Structured VAE
* [Composing graphical models with neural networks for structured representations and fast inference](https://arxiv.org/pdf/1603.06277.pdf)

### Old-school: The Helmholtz Machine
* [Density Networks](https://pdfs.semanticscholar.org/8734/b13a74765d4a78ebf15c9c38991a5302d71c.pdf)
* [The Helmholtz Machine](http://www.gatsby.ucl.ac.uk/~dayan/papers/hm95.pdf)

## Black box VI
* `Done` [Black Box Variational Inference](https://arxiv.org/pdf/1401.0118.pdf)

## Automatic differentiation VI
* [Automatic Differentiation Variational Inference](https://arxiv.org/pdf/1603.00788.pdf)

## Reparametrization
* [The Generalized Reparameterization Gradient](https://arxiv.org/pdf/1610.02287.pdf)
* [Doubly Stochastic Variational Bayes for non-Conjugate Inference](http://www2.aueb.gr/users/mtitsias/papers/titsias14.pdf)

## Normalizing flows
Not sure yet what this is...

## Background
* Variational Bayes: [Beal, 2003](https://www.cse.buffalo.edu//faculty/mbeal/papers/beal03.pdf)
* Deep Generative Models: [Kingma, 2017](https://pure.uva.nl/ws/files/17891313/Thesis.pdf)


# Automatic differentiation
* [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/pdf/1502.05767.pdf)
* A demonstration with Python's Autograd: [Black-Box Stochastic Variational Inference in Five Lines of Python](http://people.seas.harvard.edu/~dduvenaud/papers/blackbox.pdf)


# Software
## Edward
* [Edward](http://edwardlib.org/)
* Paper: [Edward: A library for probabilistic modeling, inference, and criticism](https://arxiv.org/pdf/1610.09787.pdf)

## Pyro
Great tutorials!
* [Pyro](http://pyro.ai/examples/index.html)


# Inspiration
## Applications in Digital Humanities
* Cornell course on [Text mining for History and Literature](https://mimno.infosci.cornell.edu/info3350/)
