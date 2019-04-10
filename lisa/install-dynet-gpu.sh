#!/usr/bin/env bash

# Install Dynet with GPU support on Lisa.

# This is what I did and this is what worked (24 January 2019) on Lisa
# Adapted from https://dynet.readthedocs.io/en/latest/python.html#manual-installation

# This assumes you already run install-dynet-cpu.sh (downloaded dynet and eigen)

# These are necessary!
module load Python/3.6.3-foss-2017b
module load CUDA/9.0.176

cd $HOME

mkdir -p $HOME/envs/dynet-gpu
python -m venv --system-site-packages envs/dynet-gpu

source $HOME/envs/dynet-gpu/bin/activate

mkdir dynet-gpu
cd dynet-gpu
git clone https://github.com/clab/dynet.git
hg clone https://bitbucket.org/eigen/eigen -r b2e267d  # -r NUM specified a known working revision
cd dynet
mkdir build
cd build
cmake .. -DEIGEN3_INCLUDE_DIR=../../eigen -DPYTHON=`which python` -DBACKEND=cuda
# ## Returned ##
# -- The CXX compiler identification is GNU 6.4.0
# -- Check for working CXX compiler: /hpc/eb/compilerwrappers/compilers/c++
# -- Check for working CXX compiler: /hpc/eb/compilerwrappers/compilers/c++ -- works
# -- Detecting CXX compiler ABI info
# -- Detecting CXX compiler ABI info - done
# -- Detecting CXX compile features
# -- Detecting CXX compile features - done
# -- Optimization level: fast
# -- BACKEND: cuda
# -- Looking for pthread.h
# -- Looking for pthread.h - found
# -- Looking for pthread_create
# -- Looking for pthread_create - not found
# -- Looking for pthread_create in pthreads
# -- Looking for pthread_create in pthreads - not found
# -- Looking for pthread_create in pthread
# -- Looking for pthread_create in pthread - found
# -- Found Threads: TRUE
# -- Found CUDA: /hpc/eb/Debian9/CUDA/9.0.176 (found version "9.0")
# CUDA_LIBRARIES: /hpc/eb/Debian9/CUDA/9.0.176/lib64/libcudart_static.a;-lpthread;dl;/usr/lib/x86_64-linux-gnu/librt.so;/hpc/eb/Debian9/CUDA/9.0.176/lib64/libcurand.so
# -- Failed to find CUDNN in path: /hpc/eb/Debian9/CUDA/9.0.176 (Did you set CUDNN_ROOT properly?)
# -- CUDNN not found, some dependent functionalities will be disabled
# -- Eigen dir is /home/daanvans/dynet-gpu/eigen
# -- Found Cython version 0.27.1
#
# --- CUDA: CUBLAS: /hpc/eb/Debian9/CUDA/9.0.176/lib64/libcublas.so;/hpc/eb/Debian9/CUDA/9.0.176/lib64/libcublas_device.a RT: /hpc/eb/Debian9/CUDA/9.0.176/lib64/libcudart_static.a;dl;/usr/lib/x86_64-linux-gnu/librt.so;/hpc/eb/Debian9/CUDA/9.0.176/lib64/libcurand.so
# CMAKE_INSTALL_PREFIX="/usr/local"
# PROJECT_SOURCE_DIR="/home/daanvans/dynet-gpu/dynet"
# PROJECT_BINARY_DIR="/home/daanvans/dynet-gpu/dynet/build"
# LIBS="/hpc/eb/Debian9/CUDA/9.0.176/lib64/libcudart_static.a\;dl\;/usr/lib/x86_64-linux-gnu/librt.so\;/hpc/eb/Debian9/CUDA/9.0.176/lib64/libcurand.so\;-lpthread"
# EIGEN3_INCLUDE_DIR="/home/daanvans/dynet-gpu/eigen"
# MKL_LINK_DIRS=""
# WITH_CUDA_BACKEND="1"
# CUDA_RT_FILES="libcudart_static.a\;dl\;librt.so\;libcurand.so"
# CUDA_RT_DIRS="/hpc/eb/Debian9/CUDA/9.0.176/lib64\;/usr/lib/x86_64-linux-gnu"
# CUDA_CUBLAS_FILES="libcublas.so\;libcublas_device.a"
# CUDA_CUBLAS_DIRS="/hpc/eb/Debian9/CUDA/9.0.176/lib64"
# MSVC=""
# -- Configuring done
# -- Generating done
# -- Build files have been written to: /home/daanvans/dynet-gpu/dynet/build
make -j 2
cd python
python ../../setup.py build --build-dir=.. --skip-build install --user
