#!/usr/bin/env bash

OBW=data/unlabeled/news.en-00001-of-00100

echo "Processing one-billion-word dataset..."
python scripts/process-obw.py \
  "${OBW}" > "${OBW}.processed"

# make tokenization consistent with the style of the PTB
sed -i "" "s/n 't/ n't/g" "${OBW}.processed"

echo "Results saved in '${OBW}.processed'."
