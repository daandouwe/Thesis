#!/usr/bin/env bash

set -x # echo on

mkdir -p \
    train \
    dev \
    chapters

./process-alice.py

./parse-alice.py

# Make training set from chapters 2-12.
cat alice.{2..12}.txt > train/alice.train.txt
cat alice.{2..12}.toks > train/alice.train.toks
cat alice.{2..12}.tokn > train/alice.train.tokn
cat alice.{2..12}.trees > train/alice.train.trees

# Make training set from chapter 1.
cat alice.1.txt > dev/alice.dev.txt
cat alice.1.toks > dev/alice.dev.toks
cat alice.1.tokn > dev/alice.dev.tokn
cat alice.1.trees > dev/alice.dev.trees

# Trees must also be in tmp.
cp train/alice.train.trees tmp/train/alice.train.trees
cp dev/alice.dev.trees tmp/dev/alice.dev.trees
cp dev/alice.dev.trees tmp/test/alice.test.trees

# Move all chapters.
mv alice.{0..12}.* chapters

# Get oracles and vocabularies for RNNG parser.
./alice2rnng.sh
