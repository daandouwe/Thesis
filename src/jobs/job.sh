#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=24:00:00

# Loading modules
module load eb
module load Python/3.6.3-foss-2017b
module load CUDA/9.0.176
module load cuDNN/7.0.5-CUDA-9.0.176

set -x # echo on
SRCDIR=$HOME/thesis/src
DATADIR=$HOME/thesis/tmp
OUTPUT_DIR=output_dir

# Copy training data to scratch
cp -r $DATADIR $TMPDIR

# Create output directories on scratch
mkdir -p $TMPDIR/$OUTPUT_DIR/log
mkdir -p $TMPDIR/$OUTPUT_DIR/checkpoints
mkdir -p $TMPDIR/$OUTPUT_DIR/out

# Execute the Python program with data from the scratch directory
python $SRCDIR/main.py train --data $TMPDIR/tmp --root $TMPDIR/$OUTPUT_DIR --epochs 4 --lr 1e-4

# Copy output directory from scratch to home
cp -r $TMPDIR/$OUTPUT_DIR/log/* $SRCDIR/log
cp -r $TMPDIR/$OUTPUT_DIR/checkpoints/* $SRCDIR/checkpoints
cp -r $TMPDIR/$OUTPUT_DIR/out/* $SRCDIR/out
