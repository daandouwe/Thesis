#!/bin/sh

PTB_IN=../data/ptb/con/treebank3/parsed/mrg/wsj
PTB_OUT=../tmp/ptb
MAX_LINES=100

mkdir ../tmp

echo Transforming PTB into one file at $PTB_OUT.txt
python3 transform_ptb.py $PTB_IN $MAX_LINES > $PTB_OUT.txt

echo Extracting oracle transitions into $PTB_OUT.oracle
python3 get_oracle.py $PTB_OUT.txt $PTB_OUT.txt > $PTB_OUT.oracle

echo Turning oracle into configurations at $PTB_OUT.configs
python3 get_configs.py $PTB_OUT.oracle > $PTB_OUT.configs
