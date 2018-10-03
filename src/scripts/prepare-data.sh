#!/usr/bin/env bash
THESIS_DIR=$HOME/Documents/Logic/Thesis
DATA_DIR=$HOME/data/ptb/con/treebank3/parsed/mrg/wsj
OUTDIR=$THESIS_DIR/data
NAME=ptb

set -x # echo on

# make ouput folders
mkdir -p \
    $OUTDIR/train \
    $OUTDIR/dev \
    $OUTDIR/test \
    $OUTDIR/vocab/lower \
    $OUTDIR/vocab/unked \
    $OUTDIR/vocab/original

# from mrg files to linearized one-tree-per-line and train/dev/test splits.
python transform_ptb.py \
    --indir $DATA_DIR \
    --outdir $OUTDIR \
    --name $NAME

# remove traces from trees
treetools/treetools transform \
    $OUTDIR/train/$NAME.train.trees \
    $OUTDIR/train/$NAME.train.trees.notrace \
    --trans ptb_delete_traces --src-format brackets --dest-format brackets
treetools/treetools transform \
    $OUTDIR/dev/$NAME.dev.trees \
    $OUTDIR/dev/$NAME.dev.trees.notrace \
    --trans ptb_delete_traces --src-format brackets --dest-format brackets
treetools/treetools transform \
    $OUTDIR/test/$NAME.test.trees \
    $OUTDIR/test/$NAME.test.trees.notrace \
    --trans ptb_delete_traces --src-format brackets --dest-format brackets

# add back the damm space between brackets that treetools removed...
python add_space.py $OUTDIR/train/$NAME.train.trees.notrace
python add_space.py $OUTDIR/dev/$NAME.dev.trees.notrace
python add_space.py $OUTDIR/test/$NAME.test.trees.notrace

# simplify the nonterminals
python simplify_nonterminals.py $OUTDIR/train/$NAME.train.trees.notrace
python simplify_nonterminals.py $OUTDIR/dev/$NAME.dev.trees.notrace
python simplify_nonterminals.py $OUTDIR/test/$NAME.test.trees.notrace

# overwrite the old trees with the new notrace trees
mv $OUTDIR/train/$NAME.train.trees.notrace $OUTDIR/train/$NAME.train.trees
mv $OUTDIR/dev/$NAME.dev.trees.notrace $OUTDIR/dev/$NAME.dev.trees
mv $OUTDIR/test/$NAME.test.trees.notrace $OUTDIR/test/$NAME.test.trees

# get discriminative oracles
python get_oracle.py \
    $OUTDIR/train/$NAME.train.trees \
    $OUTDIR/train/$NAME.train.trees \
    > $OUTDIR/train/$NAME.train.oracle
python get_oracle.py \
    $OUTDIR/train/$NAME.train.trees \
    $OUTDIR/dev/$NAME.dev.trees \
    > $OUTDIR/dev/$NAME.dev.oracle
python get_oracle.py \
    $OUTDIR/train/$NAME.train.trees \
    $OUTDIR/test/$NAME.test.trees \
    > $OUTDIR/test/$NAME.test.oracle

# make vocabularies for lower, original and unked
python get_vocab.py \
    $OUTDIR/train/$NAME.train.oracle \
    $OUTDIR/dev/$NAME.dev.oracle \
    $OUTDIR/test/$NAME.test.oracle \
    --name $NAME \
    --textline lower \
    --outdir $OUTDIR/vocab/lower
python get_vocab.py \
    $OUTDIR/train/$NAME.train.oracle \
    $OUTDIR/dev/$NAME.dev.oracle \
    $OUTDIR/test/$NAME.test.oracle \
    --name $NAME \
    --textline unked \
    --outdir $OUTDIR/vocab/unked
python get_vocab.py \
    $OUTDIR/train/$NAME.train.oracle \
    $OUTDIR/dev/$NAME.dev.oracle \
    $OUTDIR/test/$NAME.test.oracle \
    --name $NAME \
    --textline original \
    --outdir $OUTDIR/vocab/original
