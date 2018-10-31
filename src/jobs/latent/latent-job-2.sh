#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=14:00:00

# Loading modules
module load eb
module load Python/3.6.3-foss-2017b

# Home folders
SRCDIR=$HOME/thesis/src
DATADIR=$HOME/thesis/data
EXP_DIR=$SRCDIR/lisa-experiments
EXP_NAME=latent-cpu-2

# Scratch folders
TMP=$TMPDIR/daandir
OUTDIR=$TMP/results

# General training settings
MAX_TIME=$((13 * 3600))  # in seconds (related to -lwalltime)
MAX_EPOCHS=100
MAX_LINES=-1
PRINT_EVERY=10
EVAL_EVERY=100

# Copy training data to scratch
mkdir -p $TMP/data
cp -r $DATADIR/* $TMP/data
# Create output directories on scratch
mkdir -p $OUTDIR
# Just checking if all folders are constructed correctly
ls -l $TMP

# Experiment
OPTIM=adam
LR=0.001
BATCH_SIZE=32
# Name of experiment.
NAME=${OPTIM}_lr${LR}_batch_size${BATCH_SIZE}
# Make output directory.
mkdir -p $OUTDIR/$NAME
# Run.
python $SRCDIR/main.py latent disc \
    --data $TMP/data \
    --disable-subdir \
    --logdir $OUTDIR/$NAME \
    --checkdir $OUTDIR/$NAME \
    --outdir $OUTDIR/$NAME \
    --max-epochs $MAX_EPOCHS \
    --print-every $PRINT_EVERY \
    --eval-every $EVAL_EVERY \
    --optimizer $OPTIM \
    --lr $LR \
    > $OUTDIR/$NAME/terminal.txt

echo 'Finished training. Copying files from scratch...'
# Copy output directory from scratch to home
mkdir -p $EXP_DIR/$EXP_NAME
cp -r $OUTDIR/* $EXP_DIR/$EXP_NAME

# Cleanup scracth.
rm -r $TMP/*
