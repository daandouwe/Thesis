# Notes 31 July

## Data
* Look into preprocessing: https://github.com/wmaier/treetools. We need to remove entries e.g. `(-NONE- *T*-2)`.
* Do we predict tags? Discriminative: NO. Generative: No.
* ...

## Model
- [x] Sum over losses per sentence. Not average!

## Training
* Single node has 16 processors

## Objectives:
- [x] Clean PTB data conform Dyers (?)
- [x] Simplify nonterminals
- [x] Get torch on lisa working
- [x] Update to torch 4
- [x] Match hyperparamters (keep Adam optimizer)
- [x] Glorot init for params
- [x] Get glove vectors to work well.
- [ ] Get distributed training to work
- [ ] Average over batches more stable training?
- [x] Hopefully: first 80+ F score?
