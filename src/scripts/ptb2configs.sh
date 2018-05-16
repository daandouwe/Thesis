#!/bin/sh

THESIS_DIR=../..
SRC_DIR=$THESIS_DIR/src
PTB_IN=$THESIS_DIR/data/ptb/con/treebank3/parsed/mrg/wsj
PTB_OUT=$THESIS_DIR/tmp/ptb
MAX_LINES=10000

set -x #echo on

python3 $SRC_DIR/transform_ptb.py $PTB_IN > $PTB_OUT.txt
# python3 $SRC_DIR/transform_ptb.py $PTB_IN $MAX_LINES > $PTB_OUT.txt

python3 $SRC_DIR/get_oracle.py $PTB_OUT.txt $PTB_OUT.txt > $PTB_OUT.oracle

python3 $SRC_DIR/get_configs.py $PTB_OUT.oracle > $PTB_OUT.configs
