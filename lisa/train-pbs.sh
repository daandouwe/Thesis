#!/bin/bash

# we need to get the GPU ID in the first argument, all else goes to the parser
GPU_ID=$1
PROJECT=$2
PARSER_ARGS="${@:3}"

export LANGUAGE="en_US.UTF-8"
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export PYTHONIOENCODING="UTF-8"
export PYTHONUNBUFFERED="1"
export PYTHONPATH="$PYTHONPATH:${HOME}/git/parser"

# enter virtual environment
source ${HOME}/envs/parser/bin/activate

JOB_ID_NUMERIC=${PBS_JOBID%%.*}
OUTPUT_DIR="${HOME}/models/parser/${PROJECT}/${JOB_ID_NUMERIC}_${GPU_ID}"
mkdir -p ${OUTPUT_DIR}

TIMESTAMP=`date +%s`
echo "`date` Training starts"

which python
python --version

CUDA_VISIBLE_DEVICES=${GPU_ID} python -m parser --output_dir ${OUTPUT_DIR} ${PARSER_ARGS[@]} &>> ${OUTPUT_DIR}/train.log

wait

echo "`date` Training done"

sleep 900
echo "`date` Sleep done"
