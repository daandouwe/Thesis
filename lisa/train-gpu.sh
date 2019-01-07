#!/bin/bash

# we the seed in the first argument, the output dir in the second, and all else goes to the argument parser
GPU=$(($1))
SEED=$(($2))
OUTPUT_DIR=$3
PARSER_ARGS="${@:4}"

export LANGUAGE="en_US.UTF-8"
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export PYTHONIOENCODING="UTF-8"
export PYTHONUNBUFFERED="1"

module load Python/3.6.3-foss-2017b

# activate gpu environment
source ${HOME}/envs/dynet-gpu/bin/activate

TIMESTAMP=`date +%s`
echo "`date` Training starts"

which python
python --version

python src/main.py train --dynet-autobatch 1 --dynet-devices GPU:${GPU} --dynet-seed ${SEED} --numpy-seed ${SEED} ${PARSER_ARGS[@]} &>> ${OUTPUT_DIR}/train${SEED}.log &

wait

echo "`date` Training done"

# sleep 900
sleep 2
echo "`date` Sleep done"
