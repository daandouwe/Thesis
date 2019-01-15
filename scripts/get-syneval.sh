#!/usr/bin/env bash

echo "Downloading syneval dataset..."
git clone https://github.com/BeckyMarvin/LM_syneval data/syneval

echo "Converting syneval dataset..."
mkdir data/syneval/data/converted
python scripts/convert-syneval.py

echo "Done."
