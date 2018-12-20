#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=80:00:00

# Loading modules
module load eb
module load Python/3.6.3-foss-2017b

# Home folders
SRCDIR=$HOME/thesis/dynet-rnng
DATADIR=$HOME/thesis/data
EXP_DIR=$SRCDIR/experiments/lisa
EXP_NAME=semisup-baseline=argmax+mlp

# Scratch folders
TMP=$TMPDIR/daandir
OUTDIR=$TMP/results

# General training settings
MAX_TIME=$((70 * 3600))  # in seconds (related to -lwalltime)
MAX_EPOCHS=100
MAX_LINES=-1
PRINT_EVERY=10
EVAL_EVERY=20000

# Copy training data to scratch
mkdir -p $TMP/data
cp -r $DATADIR/* $TMP/data
# Copy models embeddings to scratch
mkdir -p $TMP/models/joint
mkdir -p $TMP/models/posterior
cp -r $SRCDIR/checkpoints/joint/* $TMP/models/joint
cp -r $SRCDIR/checkpoints/posterior/* $TMP/models/posterior
# Create output directories on scratch
mkdir -p $OUTDIR
# Just checking if all folders are constructed correctly
ls -l $TMP
ls -l $TMP/data
ls -l $TMP/models

export MKL_NUM_THREADS=1

OPTIM=adam
LR=0.001
BATCH_SIZE=10
# Name of experiment.
NAME=${OPTIM}_lr${LR}_batch_size${BATCH_SIZE}
# Make output directory.
mkdir -p $OUTDIR/$NAME
# Run.
python $SRCDIR/main.py semisup disc \
    --train-path $TMP/data/train/ptb.train.trees \
    --dev-path $TMP/data/dev/ptb.dev.trees \
    --test-path $TMP/data/test/ptb.test.trees \
    --unlabeled-path $TMP/data/unlabeled/news.en-00001-of-00100 \
    --joint-model-path $TMP/models/joint \
    --post-model-path $TMP/models/posterior \
    --disable-subdir \
    --logdir $OUTDIR/$NAME \
    --checkdir $OUTDIR/$NAME \
    --outdir $OUTDIR/$NAME \
    --max-lines $MAX_LINES \
    --max-time $MAX_TIME \
    --max-epochs $MAX_EPOCHS \
    --print-every $PRINT_EVERY \
    --eval-every $EVAL_EVERY \
    --eval-at-start \
    --optimizer $OPTIM \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    --use-argmax-baseline \
    --use-mlp-baseline \
    --dynet-mem  5000 \
    > $OUTDIR/$NAME/terminal.txt


echo 'Finished training. Copying files from scratch...'
# Copy output directory from scratch to home
mkdir -p $EXP_DIR/$EXP_NAME
cp -r $OUTDIR/* $EXP_DIR/$EXP_NAME

# Cleanup scracth.
rm -r $TMP/*
