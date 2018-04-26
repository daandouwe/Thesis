# RNNG

Code for the RNNG.

## Data

We use the Penn Treebank. For now, we are only using the WSJ section.

Run the following command to convert the WSJ part of the Penn Treebank from `mrg` files to linearized, stripped, line-by-line format:
```
python transform_ptb.py > train.txt
```
Alternatively, use some integer `n` to limit the number of `mrg` files while developping:
```
python transform_ptb.py 10 > train.txt
```

Then call
```
python get_oracle.py [training file] [training file] > train.oracle
```
to extract the sequence of transitions for the discriminative parser (code taken from the [original RNNG implementation](https://github.com/clab/rnng)).
