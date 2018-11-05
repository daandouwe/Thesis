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
- [ ] Add patience to learning rate annealing
- [ ] Implement oracles from scratch
- [ ] Training from tree-file (oracle extraction in between)
- [ ] Implement dynamic oracle and training with exploration
- [ ] Implement UNKing scheme myself.
- [ ] load_checkpoint not working for batch-size 1 (?!)

Dynet
- [ ] Figure how to disable dropout...
- [ ] CPU not using all processors, why? (Only one on Lisa!)
- [ ] Have Dynet installed with GPU support on Lisa

Experiments
- [ ] Run Dynet on Lisa GPU

Semisupervised
- [ ] Semisupervised training with REINFORCE gradient for unsupervised loss (by sampling trees)
- [ ] Think about baselines
- [ ]
