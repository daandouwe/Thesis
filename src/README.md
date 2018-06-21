# RNNG

Code for the RNNG.

## Data

We use the Penn Treebank. For now, we are only using the WSJ section. We haven't made the train/dev/test splits yet.

To extract trainable parse configurations from the ptb, run the `ptb2configs.sh` from the [scripts](scripts) folder.

## Run

To run the RNNG call `testing.py` with arguments `[test, train, parse]`.


## Todo

- [ ] Incorporate dropout
- [ ] Enable multilayered LSTMs
- [X] Load pre-trained vectors (e.g. Glove)
- [ ] Turn list of transitions into tree
