# Neural language models with latent syntax
Language models with syntax in guises of various kind.

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
You can list all the options with
```bash
make list
```

Alternatively, you use command line arguments:
```bash
python src/main.py train --parser-type=disc-rnng --model-path-base=models/disc-rnng
```
For all available options use
```bash
python src/main.py --help
```


# TODO
- [ ] With new data: either add `TOP` label to predictions, or remove `TOP` from gold trees.
