# Latent variables for Recurrent Neural Network Grammars
Language models with trees as latent variables.

## Setup
Use `make` to obtain the data and install EVALB:
```bash
make data    # download ptb and unlabeled data
make evalb   # install EVALB
```

## Usage
Use `make` to train a number of standard models:
```bash
make disc        # train discriminative rnng
make gen         # train generative rnng
make crf         # train crf
make disc-vocab  # train discriminative rnng with shared ptb/unlabeled vocabulary
```
For more information, see the makefile.

You also use command line arguments:
```bash
python src/main.py train --parser-type=disc-rnng --model-path-base=models/disc-rnng
```
To get a list of all available flags use
```bash
python src/main.py --help
```
