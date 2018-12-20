#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=90:00:00

# Loading modules
module load eb
module load Python/3.6.3-foss-2017b

# Home folders
SRCDIR=$HOME/thesis/dynet-rnng
DATADIR=$HOME/thesis/data
EXP_DIR=$SRCDIR/experiments/lisa
EXP_NAME=semisup

# Scratch folders
TMP=$TMPDIR/daandir
OUTDIR=$TMP/results

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

# General training settings
MAX_TIME=$((80 * 3600))  # in seconds (related to -lwalltime)
MAX_EPOCHS=100
MAX_LINES=-1
PRINT_EVERY=10
EVAL_EVERY=20000

OPTIM=adam
LR=0.001
BATCH_SIZE=10

TRAIN=$TMP/data/train/ptb.train.trees
DEV=$TMP/data/dev/ptb.dev.trees
TEST=$TMP/data/test/ptb.test.trees
UNLABELED=$TMP/data/unlabeled/news.en-00001-of-00100

JOINT_MODEL=$TMP/models/joint
POST_MODEL=$TMP/models/posterior

MEM=5000


# 1.
NAME=semisup-nobaseline
mkdir -p $OUTDIR/$NAME
python $SRCDIR/main.py semisup disc \
    --train-path $TRAIN \
    --dev-path $DEV \
    --test-path $TEST \
    --unlabeled-path $UNLABELED \
    --joint-model-path $JOINT_MODEL \
    --post-model-path $POST_MODEL \
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
    > $OUTDIR/$NAME/terminal.txt &


# 2.
NAME=semisup-baseline=argmax
mkdir -p $OUTDIR/$NAME
python $SRCDIR/main.py semisup disc \
    --train-path $TRAIN \
    --dev-path $DEV \
    --test-path $TEST \
    --unlabeled-path $UNLABELED \
    --joint-model-path $JOINT_MODEL \
    --post-model-path $POST_MODEL \
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
    --use-argmax-baseline \
    > $OUTDIR/$NAME/terminal.txt &


# 3.
NAME=semisup-baseline=mlp
mkdir -p $OUTDIR/$NAME
python $SRCDIR/main.py semisup disc \
    --train-path $TRAIN \
    --dev-path $DEV \
    --test-path $TEST \
    --unlabeled-path $UNLABELED \
    --joint-model-path $JOINT_MODEL \
    --post-model-path $POST_MODEL \
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
    --use-mlp-baseline \
    > $OUTDIR/$NAME/terminal.txt &

# 4.
NAME=semisup-baseline=argmax+mlp
mkdir -p $OUTDIR/$NAME
python $SRCDIR/main.py semisup disc \
    --train-path $TRAIN \
    --dev-path $DEV \
    --test-path $TEST \
    --unlabeled-path $UNLABELED \
    --joint-model-path $JOINT_MODEL \
    --post-model-path $POST_MODEL \
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
    --use-argmax-baseline \
    --use-mlp-baseline \
    > $OUTDIR/$NAME/terminal.txt &

wait
echo 'Finished training. Copying files from scratch...'
# Copy output directory from scratch to home
mkdir -p $EXP_DIR/$EXP_NAME
cp -r $OUTDIR/* $EXP_DIR/$EXP_NAME

# Cleanup scratch.
rm -r $TMP/*
