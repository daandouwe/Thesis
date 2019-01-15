#!/usr/bin/env bash

mkdir data/bootleg-ptb

echo "Downloading the Penn Treebank..."
wget https://raw.githubusercontent.com/jhcross/span-parser/master/data/02-21.10way.clean -P data/bootleg-ptb
wget https://raw.githubusercontent.com/jhcross/span-parser/master/data/22.auto.clean -P data/bootleg-ptb
wget https://raw.githubusercontent.com/jhcross/span-parser/master/data/23.auto.clean -P data/bootleg-ptb

echo "Done."
