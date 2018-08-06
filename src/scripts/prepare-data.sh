#!/usr/bin/env bash
THESIS_DIR=$HOME/Documents/Logic/Thesis
PTB=$HOME/data/ptb/con/treebank3/parsed/mrg/wsj
TMP=$THESIS_DIR/tmp
MAX_LINES=-1 # no upper limit

set -x # echo on

# make ouput folders
mkdir -p \
    $TMP/train \
    $TMP/dev \
    $TMP/test \
    $TMP/vocab/lower \
    $TMP/vocab/unked \
    $TMP/vocab/upper \

# from mrg files to linearized one-tree-per-line and train/dev/test splits.
python transform_ptb.py \
    --in_path $PTB \
    --out_path $TMP
    --nlines $MAX_LINES \

# remove traces from trees
treetools/treetools transform \
    $TMP/train/ptb.train.trees \
    $TMP/train/ptb.train.trees.notrace \
    --trans ptb_delete_traces --src-format brackets --dest-format brackets
treetools/treetools transform \
    $TMP/dev/ptb.dev.trees \
    $TMP/dev/ptb.dev.trees.notrace \
    --trans ptb_delete_traces --src-format brackets --dest-format brackets
treetools/treetools transform \
    $TMP/test/ptb.test.trees \
    $TMP/test/ptb.test.trees.notrace \
    --trans ptb_delete_traces --src-format brackets --dest-format brackets

# add back the damm space between brackets that treetools removed...
python add_space.py $TMP/train/ptb.train.trees.notrace
python add_space.py $TMP/dev/ptb.dev.trees.notrace
python add_space.py $TMP/test/ptb.test.trees.notrace

# simplify the nonterminals
python simplify_nonterminals.py $TMP/train/ptb.train.trees.notrace
python simplify_nonterminals.py $TMP/dev/ptb.dev.trees.notrace
python simplify_nonterminals.py $TMP/test/ptb.test.trees.notrace

# get oracles
python get_oracle.py \
    $TMP/train/ptb.train.trees.notrace \
    $TMP/train/ptb.train.trees.notrace \
    > $TMP/train/ptb.train.oracle
python get_oracle.py \
    $TMP/train/ptb.train.trees.notrace \
    $TMP/dev/ptb.dev.trees.notrace \
    > $TMP/dev/ptb.dev.oracle
python get_oracle.py \
    $TMP/train/ptb.train.trees.notrace \
    $TMP/test/ptb.test.trees.notrace \
    > $TMP/test/ptb.test.oracle

# make vocabularies for lower, upper and unked
python get_vocab.py \
    --oracle_dir $TMP \
    --out_dir $TMP/vocab/lower \
    --textline lower
python get_vocab.py \
    --oracle_dir $TMP \
    --out_dir $TMP/vocab/unked \
    --textline unked
python get_vocab.py \
    --oracle_dir $TMP \
    --out_dir $TMP/vocab/upper \
    --textline upper
