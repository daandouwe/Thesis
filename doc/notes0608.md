# Generative model

## Barber's approximation
* Two approximations: bernoulli and importance:
  * use importance sampling
  * approx. sampling without replacement: with replacement, then discard reps.
  *

## Training
* keep trying parallel: batches 16, 32, 48 (CPU), 4, 8, 12 (GPU).
* nvidia-smi
* ssh daanvans@login-gpu.lisa.surfsara.nl -> ssh <name-node>
* use `&` to run multiple commands on the same node/processor.

# Objectives
- [ ] Still work on parallel.
- [ ] Beter use of Lisa: more GPUs per node, `&` for more computation per processor
- [ ] Some more embeddings.
- [ ] Move to generative
- [ ] Normal softmax: see if terrible
- [ ] Implement barber approx.
- [ ] Beam search
- [ ] Ancestral sampling inference.
- [ ] Investigate: how easy is shift to dependency.
