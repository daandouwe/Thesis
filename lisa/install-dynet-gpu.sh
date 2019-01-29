#!/usr/bin/env bash

# Install Dynet with GPU support on Lisa.

# This is what I did and this is what worked (24 January 2019) on Lisa
# Adapted from https://dynet.readthedocs.io/en/latest/python.html#manual-installation

# This assumes you already run install-dynet-cpu.sh (downloaded dynet and eigen)

module load Python/3.6.3-foss-2017b
module load CUDA/9.0.176

cd $HOME

mkdir -p $HOME/envs/dynet-gpu
python -m venv --system-site-packages envs/dynet-gpu

source $HOME/envs/dynet-gpu/bin/activate

# pip install --user cython  # if you don't have it already.
# mkdir dynet-base
cd dynet-base
# # getting dynet and eigen
# git clone https://github.com/clab/dynet.git
# hg clone https://bitbucket.org/eigen/eigen -r b2e267d  # -r NUM specified a known working revision
cd dynet
mkdir build-gpu
cd build-gpu
cmake .. -DEIGEN3_INCLUDE_DIR=~/dynet-base/eigen -DPYTHON=`which python` -DBACKEND=cuda

make -j 2
cd python
python ../../setup.py build --build-dir=.. --skip-build install --user
