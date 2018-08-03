#!/usr/bin/env bash

THESIS_DIR=$HOME/Documents/Logic/Thesis
SRC_DIR=$THESIS_DIR/src
PTB_IN=$HOME/data/ptb/con/treebank3/parsed/mrg/wsj
PTB_OUT=$THESIS_DIR/tmp
MAX_LINES=-1 # no upper limit
TEXTLINE="unked"

set -x # echo on

# transform mrg files into linearized one-tree-per-line folders and make train/dev/test splits.
python $SRC_DIR/transform_ptb.py --in_path $PTB_IN --nlines $MAX_LINES --out_path $PTB_OUT

# # remove traces from trees
# $SRC_DIR/scripts/treetools/treetools transform \
#           $PTB_OUT/train/ptb.train.trees $PTB_OUT/train/ptb.train.trees.notrace \
#           --trans ptb_delete_traces --src-format brackets --dest-format brackets
# $SRC_DIR/scripts/treetools/treetools transform \
#           $PTB_OUT/dev/ptb.dev.trees $PTB_OUT/dev/ptb.dev.trees.notrace \
#           --trans ptb_delete_traces --src-format brackets --dest-format brackets
# $SRC_DIR/scripts/treetools/treetools transform \
#           $PTB_OUT/test/ptb.test.trees $PTB_OUT/test/ptb.test.trees.notrace \
#           --trans ptb_delete_traces --src-format brackets --dest-format brackets
#
# # add back the damm space that treetools removed...
# python $SRC_DIR/add_space.py $PTB_OUT/train/ptb.train.trees.notrace
# python $SRC_DIR/add_space.py $PTB_OUT/dev/ptb.dev.trees.notrace
# python $SRC_DIR/add_space.py $PTB_OUT/test/ptb.test.trees.notrace

# get oracles
python $SRC_DIR/get_oracle.py \
    $PTB_OUT/train/ptb.train.trees $PTB_OUT/train/ptb.train.trees > $PTB_OUT/train/ptb.train.oracle
python $SRC_DIR/get_oracle.py \
    $PTB_OUT/train/ptb.train.trees $PTB_OUT/dev/ptb.dev.trees > $PTB_OUT/dev/ptb.dev.oracle
python $SRC_DIR/get_oracle.py \
    $PTB_OUT/train/ptb.train.trees $PTB_OUT/test/ptb.test.trees > $PTB_OUT/test/ptb.test.oracle

# simplify the nonterminals
python $SRC_DIR/simplify_nonterminals.py $PTB_OUT/train/ptb.train.oracle
python $SRC_DIR/simplify_nonterminals.py $PTB_OUT/dev/ptb.dev.oracle
python $SRC_DIR/simplify_nonterminals.py $PTB_OUT/test/ptb.test.oracle

# make vocabularies
python $SRC_DIR/get_vocab.py --oracle_path $PTB_OUT --textline $TEXTLINE
