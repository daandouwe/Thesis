#!/bin/sh

PTB_IN=../data/ptb/con/treebank3/parsed/mrg/wsj
PTB_OUT=../tmp/ptb
MAX_LINES=5000

echo Transforming PTB into one file at $PTB_OUT.txt
python transform_ptb.py $PTB_IN $MAX_LINES > $PTB_OUT.txt

echo Extracting oracle transitions into $PTB_OUT.oracle
python get_oracle.py $PTB_OUT.txt $PTB_OUT.txt > $PTB_OUT.oracle

echo Turning oracle into configurations at $PTB_OUT.configs
python get_configs.py $PTB_OUT.oracle > $PTB_OUT.configs
