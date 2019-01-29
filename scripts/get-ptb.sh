#!/usr/bin/env bash

mkdir data/ptb

echo "Downloading the Penn Treebank..."
wget https://raw.githubusercontent.com/jhcross/span-parser/master/data/02-21.10way.clean -P data/ptb
wget https://raw.githubusercontent.com/jhcross/span-parser/master/data/22.auto.clean -P data/ptb
wget https://raw.githubusercontent.com/jhcross/span-parser/master/data/23.auto.clean -P data/ptb

echo "Done."
