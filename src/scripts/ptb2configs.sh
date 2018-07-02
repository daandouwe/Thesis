#!/bin/sh

THESIS_DIR=../..
SRC_DIR=$THESIS_DIR/src
PTB_IN=$THESIS_DIR/data/ptb/con/treebank3/parsed/mrg/wsj
# PTB_OUT=$THESIS_DIR/tmp/ptb
PTB_OUT=$THESIS_DIR/tmp
MAX_LINES=-1 # No upper limit

set -x #echo on
# OLD
# python3 $SRC_DIR/transform_ptb.py $PTB_IN --nlines $MAX_LINES > $PTB_OUT.txt
#
# python3 $SRC_DIR/get_oracle.py $PTB_OUT.txt $PTB_OUT.txt > $PTB_OUT.oracle
#
# python3 $SRC_DIR/new_get_configs.py $PTB_OUT.oracle

# NEW
python3 $SRC_DIR/transform_ptb.py --in_path $PTB_IN --nlines $MAX_LINES --out_path $PTB_OUT

# train
python3 $SRC_DIR/get_oracle.py $PTB_OUT/train/ptb.train.trees $PTB_OUT/train/ptb.train.trees > $PTB_OUT/train/ptb.train.oracle
python3 $SRC_DIR/get_vocab.py $PTB_OUT/train/ptb.train.oracle

# dev
python3 $SRC_DIR/get_oracle.py $PTB_OUT/train/ptb.train.trees $PTB_OUT/dev/ptb.dev.trees > $PTB_OUT/dev/ptb.dev.oracle
python3 $SRC_DIR/get_vocab.py $PTB_OUT/dev/ptb.dev.oracle
  
# test
python3 $SRC_DIR/get_oracle.py $PTB_OUT/train/ptb.train.trees $PTB_OUT/test/ptb.test.trees > $PTB_OUT/test/ptb.test.oracle
python3 $SRC_DIR/get_vocab.py $PTB_OUT/test/ptb.test.oracle
