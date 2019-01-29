#!/usr/bin/env bash

# Install Dynet with MKL support on Lisa.

# This is what I did and this is what worked (24 January 2019) on Lisa
# Adapted from https://dynet.readthedocs.io/en/latest/python.html#manual-installation

MKL_PATH=/hpc/sw/modules/modulefiles/libraries  # find this with `module avail mkl`

module load Python/3.6.3-foss-2017b
module load mkl

cd $HOME

pip install --user cython  # if you don't have it already.
mkdir dynet-base
cd dynet-base
# getting dynet and eigen
git clone https://github.com/clab/dynet.git
hg clone https://bitbucket.org/eigen/eigen -r b2e267d  # -r NUM specified a known working revision
cd dynet
mkdir build
cd build
cmake .. -DEIGEN3_INCLUDE_DIR=$HOME/dynet-base/eigen -DPYTHON=`which python` -DMKL=$MKL_PATH

make -j 8 # number of aivallable cores
cd python
python ../../setup.py build --build-dir=.. --skip-build install --user


# To check if linking with MKL was successful (https://github.com/clab/dynet/issues/1167):
#
# $ ldd ~/dynet-base/dynet/build/python/_dynet.cpython-36m-x86_64-linux-gnu.so
#    linux-vdso.so.1 (0x00007ffcb53cf000)
#    libdynet.so => /home/daanvans/dynet-base/dynet/build/dynet/libdynet.so (0x00002b581f47e000)
#    libpython3.6m.so.1.0 => /hpc/eb/Debian9/Python/3.6.3-foss-2017b/lib/libpython3.6m.so.1.0 (0x00002b581fac9000)
#    libstdc++.so.6 => /hpc/eb/Debian9/GCCcore/6.4.0/lib64/libstdc++.so.6 (0x00002b581fe0e000)
#    libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00002b581ff9a000)
#    libgcc_s.so.1 => /hpc/eb/Debian9/GCCcore/6.4.0/lib64/libgcc_s.so.1 (0x00002b581f42e000)
#    libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00002b582029e000)
#    libmkl_rt.so => /sara/sw/fortran-intel-13.1.3/composer_xe_2013.5.192/mkl/lib/intel64/libmkl_rt.so (0x00002b582063d000)
#    libiomp5.so => /sara/sw/fortran-intel-13.1.3/composer_xe_2013.5.192/compiler/lib/intel64/libiomp5.so (0x00002b5820b4a000)
#    libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00002b5820e61000)
#    libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00002b582107e000)
#    libutil.so.1 => /lib/x86_64-linux-gnu/libutil.so.1 (0x00002b5821282000)
#    /lib64/ld-linux-x86-64.so.2 (0x00002b581f258000)
