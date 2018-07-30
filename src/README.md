# RNNG

Code for the RNNG.

## Data

We use the Penn Treebank. For now, we are only using the WSJ section. We haven't made the train/dev/test splits yet.

To extract trainable parse configurations from the ptb, run the `ptb2configs.sh` from the [scripts](scripts) folder.

## Run

Run `./main.py train`

## Todo

For the discriminative model:

Data:
- [X] Load *all* sentences from training set
- [X] Make train/dev/test splits
- [X] Load pre-trained glove vectors
  * Too many words are not in glove: 7476 when `lower` is used; 9457 when `unked` is used.

Model:
- [ ] Incorporate dropout
- [ ] Enable multilayered LSTMs
- [ ] Character level embeddings
  * See how Joost did it.

Training:
- [X] Parallel training: multi-cpu training: one loss per cpu.
  * Haven't tested yet on large machine: lisa is troubled

Prediction
- [X] Turn list of transitions into tree
- [ ] Beam search decoding

For the generative model
- [ ] Everything
