#Set job requirements
#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=5:00:00

# Loading modules
module load eb
module load python/3.5.0
module load CUDA/9.0.176
module load cuDNN/7.0.5-CUDA-9.0.176

set -x # echo on
SRCDIR=$HOME/thesis/src
DATADIR=$HOME/thesis/tmp
DATANAME=ptb
OUTPUT_DIR=output_dir

# Copy training data to scratch
cp -r $DATADIR $TMPDIR

# Create output directories on scratch
mkdir -p $TMPDIR/$OUTPUT_DIR/log
mkdir -p $TMPDIR/$OUTPUT_DIR/checkpoints
mkdir -p $TMPDIR/$OUTPUT_DIR/out

# Execute the Python program with data from the scratch directory
python3 $SRCDIR/train.py --data $TMPDIR/tmp/$DATANAME --outdir $TMPDIR/$OUTPUT_DIR

# Copy output directory from scratch to home
cp -r $TMPDIR/$OUTPUT_DIR/log/* $SRCDIR/log
cp -r $TMPDIR/$OUTPUT_DIR/checkpoints/* $SRCDIR/checkpoints
cp -r $TMPDIR/$OUTPUT_DIR/out/* $SRCDIR/out

# Clean up on scratch
rm -r $TMPDIR/$OUTPUT_DIR
