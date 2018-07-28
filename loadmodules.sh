#!/bin/bash

echo 'Loading modules.'

module load eb
module load Python/3.6.3-foss-2017b
module load CUDA/9.0.176
module load cuDNN/7.0.5-CUDA-9.0.176

# export PYTHONIOENCODING="utf8"
# export PYTHONUNBUFFERED="1"
# export LC_ALL="en_US.UTF-8"
export PYTHONPATH="$PYTHONPATH:${HOME}/git/slpl-nmt"
