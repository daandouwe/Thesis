#!/usr/bin/env bash
THESIS_DIR=$HOME/Documents/Logic/Thesis
DATA_DIR=$HOME/data/ptb/con/treebank3/parsed/mrg/wsj
TMP=$THESIS_DIR/tmp
NAME=ptb

set -x # echo on

# make ouput folders
mkdir -p \
    $TMP/train \
    $TMP/dev \
    $TMP/test \
    $TMP/vocab/lower \
    $TMP/vocab/unked \
    $TMP/vocab/original

# from mrg files to linearized one-tree-per-line and train/dev/test splits.
python transform_ptb.py \
    --indir $DATA_DIR \
    --outdir $TMP \
    --name $NAME

# remove traces from trees
treetools/treetools transform \
    $TMP/train/$NAME.train.trees \
    $TMP/train/$NAME.train.trees.notrace \
    --trans ptb_delete_traces --src-format brackets --dest-format brackets
treetools/treetools transform \
    $TMP/dev/$NAME.dev.trees \
    $TMP/dev/$NAME.dev.trees.notrace \
    --trans ptb_delete_traces --src-format brackets --dest-format brackets
treetools/treetools transform \
    $TMP/test/$NAME.test.trees \
    $TMP/test/$NAME.test.trees.notrace \
    --trans ptb_delete_traces --src-format brackets --dest-format brackets

# add back the damm space between brackets that treetools removed...
python add_space.py $TMP/train/$NAME.train.trees.notrace
python add_space.py $TMP/dev/$NAME.dev.trees.notrace
python add_space.py $TMP/test/$NAME.test.trees.notrace

# simplify the nonterminals
python simplify_nonterminals.py $TMP/train/$NAME.train.trees.notrace
python simplify_nonterminals.py $TMP/dev/$NAME.dev.trees.notrace
python simplify_nonterminals.py $TMP/test/$NAME.test.trees.notrace

# overwrite the old trees with the new notrace trees
mv $TMP/train/$NAME.train.trees.notrace $TMP/train/$NAME.train.trees
mv $TMP/dev/$NAME.dev.trees.notrace $TMP/dev/$NAME.dev.trees
mv $TMP/test/$NAME.test.trees.notrace $TMP/test/$NAME.test.trees

# get discriminative oracles
python get_oracle.py \
    $TMP/train/$NAME.train.trees \
    $TMP/train/$NAME.train.trees \
    > $TMP/train/$NAME.train.oracle
python get_oracle.py \
    $TMP/train/$NAME.train.trees \
    $TMP/dev/$NAME.dev.trees \
    > $TMP/dev/$NAME.dev.oracle
python get_oracle.py \
    $TMP/train/$NAME.train.trees \
    $TMP/test/$NAME.test.trees \
    > $TMP/test/$NAME.test.oracle

# make vocabularies for lower, original and unked
python get_vocab.py \
    $TMP/train/$NAME.train.oracle \
    $TMP/dev/$NAME.dev.oracle \
    $TMP/test/$NAME.test.oracle \
    --name $NAME \
    --textline lower \
    --outdir $TMP/vocab/lower
python get_vocab.py \
    $TMP/train/$NAME.train.oracle \
    $TMP/dev/$NAME.dev.oracle \
    $TMP/test/$NAME.test.oracle \
    --name $NAME \
    --textline unked \
    --outdir $TMP/vocab/unked
python get_vocab.py \
    $TMP/train/$NAME.train.oracle \
    $TMP/dev/$NAME.dev.oracle \
    $TMP/test/$NAME.test.oracle \
    --name $NAME \
    --textline original \
    --outdir $TMP/vocab/original
