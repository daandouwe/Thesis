# Dynet RNNG
The RNNG, implemented in dynet!

## Usage
For single sentences, use:
```bash
./main.py train disc --data ../data --batch-size 1
```
For batches, use:
```bash
./main.py train disc --data ../data --batch-size 32 --dynet-autobatch 1 --dynet-mem 3000  # more memory needed for autobatching
```
where we can use dynet's autobatching!


## TODO
From Pytorch
- [ ] Convert Decoder classes
- [X] Convert Embedding class
- [X] Include GloVe embeddings with fine-tuning and optional freezing

Training
- [ ] Implement oracles from scratch
- [ ] Implement UNKing scheme.
- [ ] Training from tree-file (oracle extraction in between)

Dynet
- [ ] Figure how to disable dropout

Experiments
- [ ] Run Dynet on Lisa GPU
- [ ] Semisupervised training with reinforce
