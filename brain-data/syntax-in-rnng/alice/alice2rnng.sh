#!/usr/bin/env bash
SCRIPTS_DIR=$HOME/Documents/Logic/Thesis/src/scripts
DATA_DIR=.
TMP=$DATA_DIR/tmp
NAME=alice

set -x # echo on

# make ouput folders
mkdir -p \
    $TMP/train \
    $TMP/dev \
    $TMP/test \
    $TMP/vocab/lower \
    $TMP/vocab/unked \
    $TMP/vocab/upper \

# get oracles
python $SCRIPTS_DIR/get_oracle.py \
    $TMP/train/$NAME.train.trees \
    $TMP/train/$NAME.train.trees \
    > $TMP/train/$NAME.train.oracle
python $SCRIPTS_DIR/get_oracle.py \
    $TMP/train/$NAME.train.trees \
    $TMP/dev/$NAME.dev.trees \
    > $TMP/dev/$NAME.dev.oracle
python $SCRIPTS_DIR/get_oracle.py \
    $TMP/train/$NAME.train.trees \
    $TMP/dev/$NAME.dev.trees \
    > $TMP/test/$NAME.test.oracle

# make vocabularies for lower, upper and unked
python $SCRIPTS_DIR/get_vocab.py \
    $TMP/train/$NAME.train.oracle \
    $TMP/dev/$NAME.dev.oracle \
    $TMP/test/$NAME.test.oracle \
    --name $NAME \
    --textline lower \
    --outdir $TMP/vocab/lower
python $SCRIPTS_DIR/get_vocab.py \
    $TMP/train/$NAME.train.oracle \
    $TMP/dev/$NAME.dev.oracle \
    $TMP/test/$NAME.test.oracle \
    --name $NAME \
    --textline unked \
    --outdir $TMP/vocab/unked
python $SCRIPTS_DIR/get_vocab.py \
    $TMP/train/$NAME.train.oracle \
    $TMP/dev/$NAME.dev.oracle \
    $TMP/test/$NAME.test.oracle \
    --name $NAME \
    --textline upper \
    --outdir $TMP/vocab/upper
