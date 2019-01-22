# Cover letter Unbabel - Summer AI Research Internship

Opening statement.

## 1. What I work on now
Currently I am writing a master's thesis under supervision of Prof. Wilker Aziz.

I investigate the question: What are effective ways of incorporating syntactic structure into neural language models?

Central in this research is a class of neural language models that explicitly model the hierarchical syntactic structure in addition to the sequence of words [Dyer et al. 2016, Buys & Blunsom 2015, 2018].

These models merge algorithms from transition based parsing ('shift-reduce parsing') adapted for joint (generative) modeling, with (recurrent) neural networks that parametrize the transition model.

The syntactic structure that decorates the words can be latent, and marginalized over, or can be given explicitly, for example as the prediction of an external parser.

Although these are fundamentally joint model, they can be evaluated as regular language models (modeling only words) by (approximate) marginalization of the syntactic structure.


In the case of the RNNG [Dyer et al. 2016], exact marginalization is intractable due to the parametrization of the statistical model, but importance sampling provides an effective approximate method. An externally trained discriminative parser is used to obtain proposal samples.

Other models provide exact marginalization, but this typically comes at the cost of a less expressive parametrization, for example one in which the features cannot be structure-dependent [Buys & Blunsom 2018].


I study the RNNG and investigate:

1. The impact of the proposal samples on the approximate marginalization. I propose a new discriminative chart-based neural parser that is trained with a global, Conditional Random Field (CRF), objective.



The parser is an adaptation of the minimal neural parser proposed in [Stern et al. 2017] which is trained with a margin-based objective. This contrast with the typical choice for a transition-based parser as proposal (a discrminatively trained RNNG).

The rationale in this research is that we posit that a globally trained model is a better proposal distribution than a locally trained transition based model. A global model has ready access to competing analyses that can be structurally dissimilar but close in probability, whereas we hypothesize that a locally trained model is prone to produce locally corrupted structures that are nearby in transition-space.

To promote more diverse samples, the transition distributions are flattened, causing as a downside for the model to visit . This is a general challenge for greedy transition based models that is typically answered to train dynamic oracles [Golberg & Nivre 2012] (also called 'exploration' [Ballesteros 2016; Stern et al. 2017], instances of imitation learning [Vlachos 2012; Eisner et al. 2012]), a direction which we do not consider in this research.

2. Semi-supervised training by including unlabeled data.

A major drawback of these syntactic language models is that they require annotated data to be trained, and preciously little of such data exists.

To make these joint models competitive language models they need to make use of the vast amounts of unlabeled data that exists.

I extend the training to the unsupervised domain by optimizing a variational lower bound on the marginal probabilities. This jointly optimizes the parameters of proposal model (also named the 'posterior' in this framework) and the joint model.

We obtain gradients for this objective using the score function estimator [Fu 2006], also known as REINFORCE [Williams 1992], which is widely used in the field of deep reinforcement learning, and we introduce an effective baseline based on argmax decoding [Rennie et al. 2017], which significantly reduces the variance in this optimization procedure.

Our CRF parser particularly excels in the role of posterior thanks the independence assumptions that allow for efficient exact computation of key quantities: the entropy term in the lower bound can be computed exactly using Inside-Outside algorithm, removing one source of variance from the gradient estimation, and the argmax decoding can be performed exactly thanks to Viterbi, making the argmax baseline even more effective.

##### 3. Alternative, simpler, models
I investigate the added value of making syntax a discrete latent variable by comparing these models with an alternative approach that trains language models jointly with a syntactic side-objective in the framework multitask learning.

Multitask learning of a neural language model with a syntactic side objective is a competitive and robust alternative method to infuse neural language models with syntactic knowledge. Training the syntactic model on data that mixes gold trees with predicted 'silver' trees for unlabeled data is a competitive and robust alternative to fully principled semi-supervised learning. We consider these alternatives in order to quantify significance of the latent structure, and the semisupervised training on the other hand, as measured by some external performance metric.

I propose a simple multitask neural language model that predicts labeled spans from the RNN hidden states, using a feature function identical identical to that used in the CRF parser. A similar strategy has recently proposed in work on semantic parsing and is called a 'syntactic scaffold' [Swayamdipta et al. 2018].

##### 4. Targeted syntactic evaluation
[TBC]



## 2. What I have worked on before
I am currently writing a thesis for the MSc in Logic at the University of Amsterdam, where I followed courses on Theoretical Computer Science, Machine Learning and Natural Language Processing. I hold a BA in Liberal Arts and Sciences from the Amsterdam University College, a joint undergraduate college by the University of Amsterdam (UvA) and the Free University Amsterdam (VU), where I studied Philosophy and Linguistics. Between the BA and the MSc I partially completed a BSc in Mathematics.

In the Master program I ventured into Theoretical Computer Science and Discrete Mathematics, after which I rejoined with my interest in language and linguistics through Machine Learning and especially Natural Language Processing (courses in the program of Artificial Intelligence).

I learned a lot from working on individual projects:

1. Latent Dirichlet Allocation with different inference methods (collapsed Gibbs sampling and (stochastic) Variational Inference, amortized Varitional Inference (using a neural network)).
2. Probabilistic machine translation with a Conditional Random Field and latent syntactic transductions (course project for NLP2).
3. Graph-based dependency parsing with different learning algorithms (LSTM features with a biaffine scoring function, and averaged structured perceptron on manually defined features)
4. Transition-based dependency parsing with a Multilayer Perceptron transition classifier.
5. Language modeling (smoothed n-grams, neural n-gram, recurrent neural networks)
6. Word embeddings in various ways (GloVe, SVD from a positive PMI weighted cooccurence matrix).
7. A number of didactic projects, implemented with teaching in mind: CKY parsing with rules and probabilities estimated from a treebank; ngram language models witg smoothing;

In my projects I make regular use of the great libraries that the Python ecosystem has to offer: Dynet and PyTorch, and earlier some Tensorflow, wherever I need neural networks; Cython and Numpy for speed and general numerical computation; multiprocessing and joblib whenever I see the opportunity for parallel computation; matplotlib, seaborn, and pandas wherever I want to inspect, manipulate and visualize data.

I strive to make these projects well-organized, well-documented and self-contained and I increasingly learn to appreciate the role of bash scripts in this: to obtain and process data in transparent ways, to standardize training and evaluation, and to promote transparency and reproducibility in the process. I believe that the code for my thesis is the culmination of this learning process.

I have experience teaching students from the MSc in Artificial Intelligence. In the first year master's course Natural Language Processing 1, I supervised student projects in which they implemented a neural network graph-based dependency parser from scratch in Python using PyTorch. This was a semester-long project with 30 students that met twice per week. I prepared and presented mini-lectures on deep learning and dependency parsing, prepared code examples in PyTorch, and co-graded the final reports. Besides the projects I contributed to the course materials by designing programming practicals which where handed out in Jupyter Notebook. In teaching I was supported by Joost Bastings and the course was taught by Tejaswini Deoskar (see academic references in CV).


## 3. Why I want to work at Unbabel




## 4. What future work interests me
