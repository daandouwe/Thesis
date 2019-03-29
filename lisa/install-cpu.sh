# This worked (19 March 2019) on Lisa
# -Daan

cd ~/thesis

source lisa/lisa-cpu.sh

pip install --user cython  # if you don't have it already.
mkdir dynet-base
cd dynet-base
# getting dynet and eigen
git clone https://github.com/clab/dynet.git
hg clone https://bitbucket.org/eigen/eigen -r b2e267d  # -r NUM specified a known working revision
cd dynet
mkdir build
cd build
cmake .. -DEIGEN3_INCLUDE_DIR=../../eigen -DPYTHON=`which python` -DMKL=TRUE
# Response:
# -- Found MKL
#    * include: /sara/sw/fortran-intel-13.1.3/composer_xe_2013.5.192/mkl/include,
#    * core library dir: /sara/sw/fortran-intel-13.1.3/composer_xe_2013.5.192/mkl/lib/intel64,
#    * compiler library: /sara/sw/fortran-intel-13.1.3/composer_xe_2013.5.192/compiler/lib/intel64/libiomp5.so
# -- Optimization level: fast
# -- BACKEND not specified, defaulting to eigen.
# -- Eigen dir is /home/daanvans/dynet-base/eigen
# -- Found Cython version 0.27.1

make -j 8 # number of aivallable cores
cd python
python ../../setup.py build --build-dir=.. --skip-build install --user

# check if MKL link was succesful:
# ldd _dynet.cpython-36m-x86_64-linux-gnu.so
# 	linux-vdso.so.1 (0x00007fffe49c6000)
# 	libdynet.so => /home/daanvans/dynet-base/dynet/build/dynet/libdynet.so (0x00002b012e3b6000)
# 	libpython3.6m.so.1.0 => /hpc/eb/Debian9/Python/3.6.3-foss-2017b/lib/libpython3.6m.so.1.0 (0x00002b012ea01000)
# 	libstdc++.so.6 => /hpc/eb/Debian9/GCCcore/6.4.0/lib64/libstdc++.so.6 (0x00002b012ed46000)
# 	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00002b012eed2000)
# 	libgcc_s.so.1 => /hpc/eb/Debian9/GCCcore/6.4.0/lib64/libgcc_s.so.1 (0x00002b012e365000)
# 	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00002b012f1d6000)
# 	libmkl_rt.so => /sara/sw/fortran-intel-13.1.3/composer_xe_2013.5.192/mkl/lib/intel64/libmkl_rt.so (0x00002b012f575000)
# 	libiomp5.so => /sara/sw/fortran-intel-13.1.3/composer_xe_2013.5.192/compiler/lib/intel64/libiomp5.so (0x00002b012fa82000)
# 	libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00002b012fd99000)
# 	libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00002b012ffb6000)
# 	libutil.so.1 => /lib/x86_64-linux-gnu/libutil.so.1 (0x00002b01301ba000)
# 	/lib64/ld-linux-x86-64.so.2 (0x00002b012e190000)
#
# YES!
