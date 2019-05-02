# March 21
The supervised models are running. They might need to run longer than 5 days.

## Resume training
Write code to resume training.
- [X] Load model from checkpoint
- [X] Load all training settings from state.json
- [X] Can we append to tensorboard writer? (NO, but it's ok.)
- [X] Overwrite the model? Or make folder new? I think overwrite (we save the originals separately). Overwrite in the same folder.
Problem: we keep training from the *best model*, and not the *latest model*. The learning is the latest, though. But: we only resume training for models that are not poperly converged, meaning the latest model is basically the best model. Phew.

## CSV data from tensorboard
- [X] Write code to convert tensorboard logs to CSV for plots.
- [X] See if we can plot this well (notebook).

## Proposal samples
- [X] See if the code to sample proposals works, use the models trained so far (on lisa!)

## Sample inspection
- [X] Find sentences with large number of samples. Or very uniform. Or low entropy. Or with different modes.

## Sample effect
- [X] Which sequence:
  (a) 1 (both argmax and sample), 2, 4, 8, 16, 32, 64, 128 (or 100)
  (b) 1 (both argmax and sample), 5, 10, 20, 50, 100
  (c) 1, 10, 25, 50, 75, 100
- [X] Number of repetitions? (15, max number)
- [X] Different temperatures:
  (a) 1.0
  (b) 0.8  

## Perplexity computation
See Buys et al. (2018) for this.
- [X] Joint
- [ ] Argmax joint (?) Is this just marginalization with only the argmax?
- [ ] Argmax conditional (?)


# March 22
## Chekckup
- [X] Are all models running correctly? (Will most likely need to continue another 5 days...)
- [X] Sampling is rerunning (something wrong with the memory... Might be horrible! Watch out for this).
- [X] Sample experiment is rerunning (stupid mistake with commented-out line).


# March 25

## Experimental
- [X] Check all experiments, did anything go wrong?
    - What happened with the test perplexity of the gen-rnng??? Apparently, something is fucked, because
  all the test-perplexities are fucked, and so something went fucked up. I DON'T KNOW WHAT! FUCK. YOU.
- [X] Determine which experiments to continue running:
    - CRF continue
- [X] Check sample experiment, did everything go well?
    - CRF didn't do anything; check error log
    - RNNG perplexity is all over the place, very reminiscent of the test perplexities
- [ ] Sample proposals for disc and crf
- [X] Run sample experiment again
- [X] Run syneval on LM

## Training
Start semisupervised training.

## Plotting
- [X] Write code to make plots of sample experiment.

## Writing
- [X] Try style of Muriel.

## Wilker
- [ ] Write wilker with the plots; ask if enough convergence.

## Home
- [X] Sample crf proposals
- [X] Sample disc proposals
- [ ] Evaluate GenRNNG


# March 26
1. Both the GenRNNG and LM-CCG seem to be fucked: the test perplexity is way higher than the dev perplexity.
2. The syneval results for LM are worse than what should be expected: rerun at the end on GPU. For now: rest and accept.
3. Samples for CRF are done. Samples for RNNG under way.

## Experiment
- [X] Rerun GenRNNG with proper proposals. Try to fix the problem of incoherent results. Running with proper samples! Done on april 2.

## Code
- [X] Rewrite sample experiment to work more efficiently (15 sets of samples, then subsample).

## Writing
- [ ] Work out all suggestions by Wilker.

## CRF
Big surprise! The entropy appears to be working... But when I dy.renew_cg() the nan appears... WTF? YES this is CORRECT this really happens. W.T.F.


# March 27

# Experiment
- [X] Sample experiment succesfully produced samples. Evaluation takes too much memory. Retry with smaller number. Fix Fscore.

# Resume
- [X] Resumes are running!! Lets try to see how they are doing.
  - Cancelled GEN because it makes no sense... We now use different proposal samples!
- [X] LM resume requires to change the trainer class.


# March 28
I have good news and bad news. Good news. First, I have converged models: disc-rnng, crf, gen-rnng, and I have them all evaluated. Second, the crf computation of the entropy works! Bad news: semisupervised training will not be possible. Simply not. One epoch on the ptb takes


# April 5
## Models etc
  - CRF all finished
  - RNNG finished Sunday 7 april

## Syneval
- [X] LM
- [X] Multitask LMs

## Notities Wilker uitwerken
- [ ] Probeer Hoofdstukken af te maken. Meer inleiding, en kijk naar de resultaten sectie. Aftikken! Maar nu even pauze.


# April 10
## Paper!
- [X] Check out this shabang new paper, what did they do??

## Semisup
1. Understand semisupervised results:
  - [X] Download models, evaluate them (?), interpret training logs to see what went wrong
2. Make changes accordingly.
  - [X] Also evaluate posterior F1
3. Unsupervised training
  - rewrite semisup training as unsupervised
  - train unsupervised with CRF! Apparently that works a whole lot better than the RNNG proposal
4. Some lessons from the unsup paper:
  - Batch size 16, K=8 samples
  - Baseline: use the mean of _all other samples_ (which makes sense for larger)
5. Speed (time for 1 epoch)
  - semisup-crf-argmax-1sample        1d9h24m36s    max 3 epochs
  - semisup-crf-argmax-1sample-exact  1d19h40m43s   max 2 epochs
  - semisup-disc-argmax-1sample       17h07m07s     max 5 epochs
  - semisup-disc-argmax-3samples      1d2h50m32s    max 5 epochs
6. Evaluate development training on much smaller datasets! Instead of the full 1700, just do 200 first sentences.

## Syneval
- [X] Download RNNG syneval results, make syneval plots.
- [X] How much influence does UNK have?
- [X] Do syneval only on sentences without UNK (see "unsupervised" paper). Write code that filters the files.

## Writing
1. CRF chapter
  - [ ] Write down explicit algorithms for inside, outside, entropy, and sampling: take more ownership! you did this fucking work!
  - [ ] ELBO: rewrite the term with the KL term as usual, and make the connection more eplicitly.
2. Make notes about preliminary experiments with fully unsupervised that collapse to trivial solutions.


# April 11

## Lisa
- [ ] Request nodes with more memory.

## Semisup
- [X] Allow Dev computation on smaller dataset.
- [X] Figure out maximum training time for semisupervised models based on above times.

## Unsupervised
- [X] Adapt semisupervised training class to allow unsupervised training on the PTB


# April 12

- [X] Supervised pre-training unlabeled
- [X] Semisupervise unlabeled training from scratch
- [ ] Run unsupervised unlabeled training on Lisa
- [ ] Alternative entropy computation, see difference
- [X] Alternative baseline: sample-mean of others

## Plan
Semisup:
  * Labeled semisup:
    - Disc, doable already running, likely faulty, but fuck it.
    - CRF absolutely undoable, mention O(n^3|L|)
  * Unlabeled semisup
    - Disc (maybe work)
    - CRF (will work) => Collapse to trivial trees.
  * Unlabeled unsup
    - Disc (Likely fail)
    - CRF (will likely work)  => Collapse to trivial trees.
Sup:
  * Pretrain unlabeled models
    - Gen
    - CRF

Plan:
  * batch size 1
  * 6 samples
  * argmax baseline
  * leave one out baseline
  * no normalization
  * scaling yes
  => total of 8 models


# April 14
- [X] Q: Why is entropy not the same as mean post-logprob? A: we were including the marginal of the impossiblem top node
- [X] Is the objective 100% correct?
- [X] Supervised unlabeled training (make sure CRF has only 2 labels!). Filter out sentences with unary chains
- [X] Semisupervised with unlabeled ptb: have a test statement when loading unlabeled


## Terrible discovery
- The CRF gives biased samples?
- Sample, then collapse dummy nodes, and disalowing the Dummy at the top introduces bias. Yes, but only because we implement entropy wrong.
- Result: the mean of the samples diverges wildly from the exact entropy.

This is how I solved it:

  * Inside
  Sum over only those trees that do not have Dummy at top.
  ```python
    if length == len(words):
        chart[left, right, self.label_vocab.values[0]] = semiring.zero()

    summed[left, right] = semiring.sums([
        chart[left, right, label]
        for label in self.label_vocab.values[start:]
    ])
  ```

  * Outside
  Exlude the entire part of the forest consisting of Dummy at the top
  ```python
  if left == 0 and right == len(words):
      for label, label_index in self.label_vocab.indices.items():
          if label_index == 0:  # dummy label
              chart[left, right, label] = semiring.zero()
          else:
              chart[left, right, label] = semiring.one()
  ```

## NOTE
1. The posterior can cheat: we are unbinarizing, and so many different trees collapse to a single one. Thus we can have a high entropy, but still all the samples could be the same when unbinarized. So we can expect that the model will collapse to single node trees.
2. This is what we observe.

# April 15
1. I observe collapse for unsup and semisup: all trees are like `(X ....)`
  - [X] Is there a problem with the entropy still? NO
  - [X] Why are all the samples of very low probability, but are the samples all the same? A: the model learns a perfect uniform distribution over all internal @ nodes, and all internal X nodes have probability 0. Therefore, all the trees get the same probability, which is still very low. I checked: all the sample trees _are_ different, before collapsing the dummy nodes. So sampling works fine, and the numbers make sense.
  - [X] Scaling with temperature (either dividing node scores, or scaling the distributions), works, and diversifies the samples just perfectly.
  - [X] The perplexity is waay too low, and that is because of the extremely skewed proposal samples. The perplexity will be much higher when we amp the temperature.

2. Pretrain the models unlabeled.
  - [X] Decoder unlabelize the proposal samples.
    * DiscRNNG and GenRNNG not much faster...
    * CRF model is _much_ faster.


# April 16
Writing

1. RNNG
  - [X] Introduction
  - [X] Composition and attention section
  - [X] Emphasize that the support of the RNNG is practically unbouded.
  - [X] Subsection of training: speed and complexity. (in background!)
  - [X] Result tables
  - [X] Entropy figures
  - [X] Perplexity figures.
  - [X] Helemaal rechtrekken.
  - [ ] Complexity in training
  - [ ] Speed in experiments, including convergence time.

2. CRF
  - [X] Introduction
  - [X] Describe time complexity.
  - [X] Results: plots, numbers
  - [X] Binarization, dummy label, impact on inside and outside, and entropy computation
  - [X] CRF has ambiguity in derivation: CRF describes distribution over 'derivations', not over 'trees'. So entropy computation is   over the derivations, _not_ over the trees: this explains the semisupervised behaviour.
    * CRF is distribution over binary trees, not over collapsed trees! So actually not porperly suitable as proposal distribution. This is not so strongly noticable when dealing with large (100+) label-set, but becomes totally untenable in case of unlabeled (1 label).
    * Also: the support of the GenRNNG is strictly greater than the support of the CRF: unbounded unary chains are not supported by CRF, only small unary chains that are in the PTB. This is not a problem really for the pretrained models; it is a problem for training from scratch.
  - [X] Check: is the MC estimate still valid, given that the CRF models derivations?

3. Semisup
  - [X] DiscRNNG: (labeled and unlabeled) from scratch impossible; from pretraining unrails completely.
  - [X] CRF: from pretraining impossible (tooo slow); from scratch looks promising, but we run into the problem of distribution.
  - [X] When move to unbinarized trees the approach is possible!
  - [X] Describe CRF as fully connected forest, then it is possible.
  - [X] Describe how to prune CRF forest.

4. Syneval
  - [X] Introduction
  - [X] Result plots.
  - [X] Describe
  - [ ] All results are rather bad, barely above random. The results reported in the Linzen paper are much better. Probably because of the small dataset.

5. Conclusion
  - [X] Everything

6. VI appendix
  - [X] Baseline section.
  - [ ] Confusion about baselines and control variates.... Not very relevant though. URNNG people write they use a "control variate". Also, confusion between E[f(X)] and E[f(X) grad_{theta} log p(X)]. Which is one are we dealing with? Clear this up.


## April 23

## Bureacracy
1. Read MoL graduation guide: is everything under control? How is my datanose status etc?
2. Get access to lisa again.

## Writing
1. Wrap up RNNG chapter, see checklist.
2. Start working on semisupervised chapter


# April 30
Notes from discussion with Wilker, and other ideas.

  - [X] To summarize our contribution in one line: introduce a CRF parser that can perform posterior inference with the additional benefit
   of able to compute key quantities.
  - [X] Introduction: joint models are nice, cite classifier image, describe p(x, y, z, ...) as the true model of the world (sentence x, with structure y, and meaning z, with speaker a, to speaker b, etc.), and that p(x) = sum_y sum_x ... p(x, y, z, ...) is the probability in isolation of the sentence. Maybe some example, p(x, speaker = Wilker) >> p(x, speaker = Daan) (maybe something like that?), and p(x) = sum_{speaker} p(x, speaker). We call y, z, ... latent variabels: they help expressiveness in the model.
  - [X] CRFs are introduced by Cullum etc. Put in description of models, or in
  - [X] Semisup Fix KL (should be KL(q || p)). (Maybe derive and say something about finding mean, but underestimating variance?)
  - [X] Put SGD part with neural networks in background.
  - [ ] CRF: In section 4.4.1 compare the training of the CRF to the training of a local model.
  - [ ] Counting semiring in CRF appendix.
  - [ ] Syneval example: say something about our RNNG and that it still prefers the ungrammatical one despite the entropy in the posterior.
  - [ ] Syneval: entropy with discriminative models is a classification method.
  - [ ] Syneval example: Draw the trees? Or otherwise write them differently. Now they are too hard too read.
  - [ ] Future work: write down ideas about hurting the RNNG, with breaking Markov assumptions. Using latent factor model for labels. Every
   time a constituent is closed, words following a closed constituent are conditionally independet of the words and subtrees in that constituent given the label.
  - [ ] Syneval figure: draw 0.5 line.
  - [ ] Syneval: look into object relatives (RNNG do well on them). Maybe change example?
  - [ ] Conclusion: (Sparsemap) keep within the framework of VI: describe that the objective is to make inference model sparser. Still do reinforce, something like that. In this case, support of posterior needs only be subset of joint. The Sparsemap solution can choose to shrink that support.


## Most important

## Intro
- [X] RNNGs are strong language models.

### Background
- [X] Trees on right page.
- [X] Call the normal form binarization and mention CRF chapter

### RNNG
- [X] Move images close into chapter.
- [X] RNNG: Reranking with p(x, y) for parsing is an approximation: we want to maximize p(y|x), but this is intractable because of sum p(y|x) =
 p(x,y) / p(x). Instead approximate. We hope that KL(q(y|x)||p(y|x)) is small (just a hope). So instead get a whole bunch of y ~ p(y|x). Then if p(x,y) is high, then p(y|x) is high. Because p(y|x) propto p(x,y). In the case of VI, we actually _optimize_ q(y|x) to be like p(y|x).

### CRF
- [X] Fix ugly ordering of figures: put everything on one page?
- [X] Add note on binarization, referring to the trees image, note that varnothing is in Lambda.
- [X] CRF: indicator notation
- [ ] FIX SPEED AND COMPLEXITY IN TRAINING
- [ ] Write logscore as cliques prod_{c in mathcal{C}} psi(a_c), or as index sets prod_{I} psi(a_I)
- [ ] Recall complexity in the training objective, and note dependence also on labelset.
- [ ] Note that speedup is possible we can speed up by training on less labels. Note though that O(n^3 |G|) = O(n^3 |Lambda|^3)!!! So cubic in the number of
- [ ] CRF: say that implementation of new inference algorithm is working, solution is easy to implement. But: no results yet (out of time).
- [ ] CRF: derivational ambiguity consequences, mention that only really problems as proposal, not really as supervised parser. Mention how Stern et al deal with this (asigning same score to all dummy labels).
- [ ] CRF: make nice point about entropy computation: the difference between the weight of all trees (log Z) and the weight of the expected tree sum_v [log psi(v) mu(v)]
- [ ] CRF: E_{p(y|x)}[log Psi(x, y)] = E_{p(y|x)}[ sum_{v in Y} log psi(x, Y)]

### Syneval
- [ ] Ugly too long trees, draw as trees? Otherwise introduce notation in background.

### Semisup
- [X] Change entropy notation to H(q(y|x)) (abusing notation)
- [X] Make L a function of lambda
- [X] Semisup: define KL divergence, explaining KL(q||p) >= 0 and 0 when q = p. Difference with EM: q is approximation to p and we cannot get KL(q||p) = 0.
- [X] Related work: the structVAE: also transition based right? Did not derail, right? Why? Say something about that.
- [X] Related work in semisupervised learning: Lapatta paper. This paper does not do VI because they do not update the posterior
 parameters (source: talk). Instead they use a pretrained posterior, that is kept fixed. The joint parameters are optimized to be close the posterior. This is not VI. Therefore, they did not run into this problem we had.

## Conclusion
- [ ] Our approach of neural joint p(x,h) + CRF posterior q(h|x) is possible for other latent structure: h tags (p(x,h) a RNN LM + tag information), linear chain CRF for q(h|x); h  dependency tree p(x,h) a generative dep parser, and a matrix tree theorem CRF for q(h|x).

# Request Wilker to look at:
  1. Optimization background (page 22)
  2. Newer RNNG definition (pages 26-27)
  3. RNNG inference small notes about proposal and true posterior (page 32)

# Response Wilker
  1. I have greatly expanded the semisup related work (with all the stuff I was actually planning but dropped). I've added Neubig (VAEs with tree-structured latent variables), and Lapatta (RNNG 'VI').
  3. I'm not sure I understand Caio's work... is it like a reparameterization trick for trees? Like fully differentiable using Gumbels softmax instead of argmax inside an Eisner inference program? But they don't obtain a _hard_ tree, only a _soft_ mixture of trees? Something like that?  And is it differentiable... or not? To avoid embarassing myself (I do not have time to properly read the paper again) I'd like to not include it...
  3. I'm sorry for Jelle that he doesn't get his deserved credit but tbh I don't see the need of citing the TreeLSTM: it is used for encoding in classification models p(c|x) with c some label like sentiment (right?). This does not remotely appear anywhere in my thesis.
  4. I have looked a little bit at the "syntax in features" work that you mention, but I'd like to keep it out of the semisup chapter. Like, the "structured attention" paper (https://arxiv.org/abs/1702.00887) looks suuuuper complicated with the Li en Eisner second order semiring stuff, but as far as I can tell it's just compting some attention weights and averaging vectors, right? Again, not really relevant I feel. I mean, I can discuss it when the questions arise in the presentation, and then I'll just say that: p(x, y1) + p(x, y2)!= p(x, y1 "+" y2) (!)
