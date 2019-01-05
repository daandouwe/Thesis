#!/usr/bin/env bash

# sets the paths to the model with highest fscore or lowest perplexity
# used in the makefile to select models for evaluation and semisup training

export CRF_PATH=models/$(ls models | grep '^crf_dev=[0-9.]\+$' | cut -c9- | sort -n | tail -n1 | sed 's|^|crf_dev=|')/model
export DISC_PATH=models/$(ls models | grep '^disc-rnng_dev=[0-9.]\+$' | cut -c15- | sort -n | tail -n1 | sed 's|^|disc-rnng_dev=|')/model
export GEN_PATH=models/$(ls models | grep '^gen-rnng_dev=[0-9.]\+$' | cut -c14- | sort -n | tail -n1 | sed 's|^|gen-rnng_dev=|')/model

export CRF_VOCAB_PATH=models/$(ls models | grep '^crf_vocab=semisup_dev=[0-9.]\+$' | cut -c23- | sort -n | tail -n1 | sed 's|^|crf_vocab=semisup_dev=|')/model
export DISC_VOCAB_PATH=models/$(ls models | grep '^disc-rnng_vocab=semisup_dev=[0-9.]\+$' | cut -c29- | sort -n | tail -n1 | sed 's|^|disc-rnng_vocab=semisup_dev=|')/model
export GEN_VOCAB_PATH=models/$(ls models | grep '^gen-rnng_vocab=semisup_dev=[0-9.]\+$' | cut -c28- | sort -n | tail -n1 | sed 's|^|gen-rnng_vocab=semisup_dev=|')/model

export LM_PATH=models/$(ls models | grep '^lm_dev=[0-9.]\+$' | cut -c8- | sort -nr | tail -n1 | sed 's|^|lm_dev=|')/model
export MULTI_LM_PATH=models/$(ls models | grep '^multitask-lm_dev=[0-9.]\+$' | cut -c18- | sort -nr | tail -n1 | sed 's|^|multitask-lm_dev=|')/model

echo "Set the following paths:"
echo "CRF_PATH=$CRF_PATH"
echo "DISC_PATH=$DISC_PATH"
echo "GEN_PATH=$GEN_PATH"
echo "CRF_VOCAB_PATH=$CRF_VOCAB_PATH"
echo "DISC_VOCAB_PATH=$DISC_VOCAB_PATH"
echo "GEN_VOCAB_PATH=$GEN_VOCAB_PATH"
echo "LM_PATH=$LM_PATH"
echo "MULTI_LM_PATH=$MULTI_LM_PATH"
