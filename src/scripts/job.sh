#Set job requirements
#PBS -S /bin/bash
#PBS -qgpu
#PBS -lwalltime=1:00:00

# Loading modules
module load eb
module load python/3.5.0
module load CUDA
# module load cudnn/8.0-v6.0

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

# Execute the Python program with data from the scratch directory
python3 $SRCDIR/train.py --data $TMPDIR/tmp/$DATANAME --outdir $TMPDIR/$OUTPUT_DIR

# Copy output directory from scratch to home
cp -r $TMPDIR/$OUTPUT_DIR/log/* $SRCDIR/log
cp -r $TMPDIR/$OUTPUT_DIR/checkpoints/* $SRCDIR/checkpoints

# Clean up on scratch
rm -r $TMPDIR/$OUTPUT_DIR
