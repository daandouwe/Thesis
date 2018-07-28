# RNNG

Code for the RNNG.

## Data

We use the Penn Treebank. For now, we are only using the WSJ section. We haven't made the train/dev/test splits yet.

To extract trainable parse configurations from the ptb, run the `ptb2configs.sh` from the [scripts](scripts) folder.

## Run

Run `./main.py train`

## Todo

- [ ] Incorporate dropout
- [ ] Enable multilayered LSTMs
- [X] Load *all* sentences from training set
- [X] Make train/dev/test splits
- [X] Load pre-trained glove vectors
  * Too many words are not in glove: 7476 when `lower` is used; 9457 when `unked` is used.
- [X] Turn list of transitions into tree
- [ ] Parallel training: multi-cpu training: one loss per cpu.
  * Very large speedups! Can work on e.g. 16 processors = batches of 16 sentences
  * Tried but not working: no backprop of loss accross CPUs
- [ ] Beam search decoding
