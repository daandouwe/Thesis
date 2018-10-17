#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=26:00:00

# Loading modules
module load eb
module load Python/3.6.3-foss-2017b

# Home folders
SRCDIR=$HOME/thesis/src
DATADIR=$HOME/thesis/data
GLOVEDIR=$HOME/embeddings/glove
EXP_DIR=$SRCDIR/lisa-experiments
EXP_NAME=disc-models-cpu-2

# Scratch folders
TMP=$TMPDIR/daandir
OUTDIR=$TMP/results

# General training settings
MAX_TIME=$((25 * 3600))  # in seconds (related to -lwalltime)
MAX_EPOCHS=100
MAX_LINES=-1
PRINT_EVERY=100
EVAL_EVERY=-1

# Copy training data to scratch
mkdir -p $TMP/data
cp -r $DATADIR/* $TMP/data
# Copy glove embeddings to scratch
mkdir -p $TMP/glove
cp -r $GLOVEDIR/* $TMP/glove
# Create output directories on scratch
mkdir -p $OUTDIR
# Just checking if all folders are constructed correctly
ls -l $TMP

# Experiment.
OPTIM=adam
LR=0.001
NUM_PROCS=16
# Name of experiment.
NAME=${OPTIM}_lr${LR}_num_procs${NUM_PROCS}_use_glove_finetune
# Make output directory.
mkdir -p $OUTDIR/$NAME
# Run.
python $SRCDIR/main.py train disc \
    --data $TMP/data \
    --disable-subdir \
    --logdir $OUTDIR/$NAME \
    --checkdir $OUTDIR/$NAME \
    --outdir $OUTDIR/$NAME \
    --max-lines $MAX_LINES \
    --max-time $MAX_TIME \
    --max-epochs $MAX_EPOCHS \
    --print-every $PRINT_EVERY \
    --eval-every $EVAL_EVERY \
    --optimizer $OPTIM \
    --lr $LR \
    --num-procs $NUM_PROCS \
    --use-glove \
    --glove-dir $TMP/glove \
    --fine-tune-embeddings \
    > $OUTDIR/$NAME/terminal.txt

echo 'Finished training. Copying files from scratch...'
# Copy output directory from scratch to home
mkdir -p $EXP_DIR/$EXP_NAME
cp -r $OUTDIR/* $EXP_DIR/$EXP_NAME

# Cleanup scracth.
rm -r $TMP/*
