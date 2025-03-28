#!/usr/bin/env bash

# this scripts sets each path to the model with highest fscore or lowest perplexity
# these variables are used in the makefile to select models for evaluation and semisup training

export CRF_PATH=models/$(ls models | grep '^crf_dev=[0-9.]\+$' | cut -c9- | sort -n | tail -n1 | sed 's|^|crf_dev=|')
export DISC_PATH=models/$(ls models | grep '^disc-rnng_dev=[0-9.]\+$' | cut -c15- | sort -n | tail -n1 | sed 's|^|disc-rnng_dev=|')
export GEN_PATH=models/$(ls models | grep '^gen-rnng_dev=[0-9.]\+$' | cut -c14- | sort -nr | tail -n1 | sed 's|^|gen-rnng_dev=|')

export CRF_VOCAB_PATH=models/$(ls models | grep '^crf_vocab=semisup_dev=[0-9.]\+$' | cut -c23- | sort -n | tail -n1 | sed 's|^|crf_vocab=semisup_dev=|')
export DISC_VOCAB_PATH=models/$(ls models | grep '^disc-rnng_vocab=semisup_dev=[0-9.]\+$' | cut -c29- | sort -n | tail -n1 | sed 's|^|disc-rnng_vocab=semisup_dev=|')
export GEN_VOCAB_PATH=models/$(ls models | grep '^gen-rnng_vocab=semisup_dev=[0-9.]\+$' | cut -c28- | sort -nr | tail -n1 | sed 's|^|gen-rnng_vocab=semisup_dev=|')

export LM_PATH=models/$(ls models | grep '^lm_dev=[0-9.]\+$' | cut -c8- | sort -nr | tail -n1 | sed 's|^|lm_dev=|')
export LM_MULTI_SPAN_PATH=models/$(ls models | grep '^lm-multitask-span_dev=[0-9.]\+$' | cut -c23- | sort -nr | tail -n1 | sed 's|^|lm-multitask-span_dev=|')
export LM_MULTI_CCG_PATH=models/$(ls models | grep '^lm-multitask-ccg_dev=[0-9.]\+$' | cut -c22- | sort -nr | tail -n1 | sed 's|^|lm-multitask-ccg_dev=|')

echo "The following variables were set:"
echo "CRF_PATH=$CRF_PATH"
echo "DISC_PATH=$DISC_PATH"
echo "GEN_PATH=$GEN_PATH"
echo "CRF_VOCAB_PATH=$CRF_VOCAB_PATH"
echo "DISC_VOCAB_PATH=$DISC_VOCAB_PATH"
echo "GEN_VOCAB_PATH=$GEN_VOCAB_PATH"
echo "LM_PATH=$LM_PATH"
echo "LM_MULTI_SPAN_PATH=$LM_MULTI_SPAN_PATH"
echo "LM_MULTI_CCG_PATH=$LM_MULTI_CCG_PATH"
