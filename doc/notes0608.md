# Generative model

## Barber's approximation
* Two approximations: bernoulli and importance:
  * use importance sampling
  * approx. sampling without replacement: with replacement, then discard reps.

## Training
* keep trying parallel: batches 16, 32, 48 (CPU), 4, 8, 12 (GPU).
* pbs_joblogin <jobnr> [nodenr]
* nvidia-smi
* how to submit to specific node in lisa?
* use `&` to run multiple commands on the same node/processor.

# Objectives
- [ ] Still work on parallel.
- [x] Refactor data and parser.
- [ ] Beter use of Lisa: more GPUs per node, `&` for more computation per processor
- [x] Some more embeddings.
- [ ] Move to generative
- [ ] Normal softmax: see if terrible
- [ ] Implement barber approx.
- [x] Beam search
- [x] Ancestral sampling inference.
- [ ] Investigate: how easy is shift to dependency.
