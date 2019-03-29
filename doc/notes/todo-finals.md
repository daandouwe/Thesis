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
