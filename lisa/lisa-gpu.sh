#!/usr/bin/env bash

# get the encoding right
export LANG="en_US.UTF-8"
export LANGUAGE="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"

# load the proper python module and cuda with which we compiled
module load Python/3.6.3-foss-2017b
module load CUDA/9.0.176
