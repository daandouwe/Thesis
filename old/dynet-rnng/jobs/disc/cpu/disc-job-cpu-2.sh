#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=25:00:00

# Loading modules
module load eb
module load Python/3.6.3-foss-2017b

# Home folders
SRCDIR=$HOME/thesis/dynet-rnng
DATADIR=$HOME/thesis/data
GLOVEDIR=$HOME/embeddings/glove
EXP_DIR=$SRCDIR/experiments/lisa
EXP_NAME=disc-models-cpu-2

# Scratch folders
TMP=$TMPDIR/daandir
OUTDIR=$TMP/results

# General training settings
MAX_TIME=$((24 * 3600))  # in seconds (related to -lwalltime)
MAX_EPOCHS=100
MAX_LINES=-1
PRINT_EVERY=10
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

export MKL_NUM_THREADS=16

# %%%%%%%%%%%%%% #
# Adam optimizer #
# %%%%%%%%%%%%%% #
for i in 1 2 3 4 5 6
do
    OPTIM=adam
    LR=0.001
    BATCH_SIZE=16
    # Name of experiment.
    NAME=${OPTIM}_lr${LR}_batch_size${BATCH_SIZE}_use_glove_seed${i}
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
        --batch-size $BATCH_SIZE \
        --use-glove \
        --glove-dir $TMP/glove \
        --dynet-autobatch 1 \
        --dynet-mem 1024 \
        --dynet-seed $((i)) \
        > $OUTDIR/$NAME/terminal.txt &
done

# %%%%%%%%%%%%% #
# SGD optimizer #
# %%%%%%%%%%%%% #
for i in 1 2 3 4 5 6
do
    OPTIM=sgd
    LR=0.1
    BATCH_SIZE=16
    # Name of experiment.
    NAME=${OPTIM}_lr${LR}_batch_size${BATCH_SIZE}_use_glove_seed${i}
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
        --batch-size $BATCH_SIZE \
        --use-glove \
        --glove-dir $TMP/glove \
        --dynet-autobatch 1 \
        --dynet-mem 1024 \
        --dynet-seed $((i)) \
        > $OUTDIR/$NAME/terminal.txt &
done


wait  # Wait for everyone to finish.

echo 'Finished training. Copying files from scratch...'
# Copy output directory from scratch to home
mkdir -p $EXP_DIR/$EXP_NAME
cp -r $OUTDIR/* $EXP_DIR/$EXP_NAME

# Cleanup scracth.
rm -r $TMP/*
