#!/bin/bash

# we the seed in the first argument, the output dir in the second, and all else goes to the argument parser
SEED=$(($1))
OUTPUT_DIR=$2
PARSER_ARGS="${@:3}"

export LANGUAGE="en_US.UTF-8"
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export PYTHONIOENCODING="UTF-8"
export PYTHONUNBUFFERED="1"
export MKL_NUM_THREADS=2

module load Python/3.6.3-foss-2017b

TIMESTAMP=`date +%s`
echo "`date` Training starts"

which python
python --version

python src/main.py train --dynet-autobatch 1 --dynet-seed ${SEED} --numpy-seed ${SEED} ${PARSER_ARGS[@]} &>> ${OUTPUT_DIR}/train${SEED}.log &

wait

echo "`date` Training done"

# sleep 900
sleep 2
echo "`date` Sleep done"
