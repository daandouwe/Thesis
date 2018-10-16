#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=12:00:00

# ============================================================================== #
# This script is an edit of `disc-job-2.sh`. Everything is the same, except
# we uses batches of size 32.
#
# This script performs 16 training experiments with the discriminative RNNG:
#
#  Adam optimizer:
#     adam_lr0.001_batch_size32
#     adam_lr0.001_batch_size32_use_glove
#     adam_lr0.0005_batch_size32
#     adam_lr0.0005_batch_size32_use_glove
#     adam_lr0.0001_batch_size32
#     adam_lr0.0001_batch_size32_use_glove
#
#  RMSprop optimizer:
#     rmsprop_lr0.001_batch_size32
#     rmsprop_lr0.001_batch_size32_use_glove
#
#  SGD optimizer:
#     sgd_lr0.1_batch_size32
#     sgd_lr0.1_batch_size32_use_glove
#     sgd_lr0.05_batch_size32
#     sgd_lr0.05_batch_size32_use_glove
#     sgd_lr0.01_batch_size32
#     sgd_lr0.01_batch_size32_use_glove
#     sgd_lr0.005_batch_size32
#     sgd_lr0.005_batch_size32_use_glove
#
#
# Learning rate values are chosen according to the settings used
# in the original paper, and additionally by considering the best
# performing values from the paper:
#    `The Marginal Value of Adaptive Gradient Methods in Machine Learning`
#
# Important: `MAX_TIME` should be a function of `-lwalltime` to make sure that
# training finishes within requested amount of time.
# Suggestion: use (number of hours in -lwalltime) - 1 hours for training.
# ============================================================================== #


# Loading modules
module load eb
module load Python/3.6.3-foss-2017b
module load CUDA/9.0.176
module load cuDNN/7.0.5-CUDA-9.0.176

# Home folders
SRCDIR=$HOME/thesis/src
DATADIR=$HOME/thesis/data
GLOVEDIR=$HOME/embeddings/glove
EXP_DIR=$SRCDIR/lisa-experiments
EXP_NAME=disc-models-3

# Scratch folders
TMP=$TMPDIR/daandir
OUTDIR=$TMP/results

# General training settings
MAX_TIME=$((11 * 3600))  # in seconds (related to -lwalltime)
MAX_EPOCHS=50
MAX_LINES=-1
PRINT_EVERY=30  # smaller because of batch-size
EVAL_EVERY=1000  # smaller because of batch-size
BATCH_SIZE=32

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


# %%%%%%%%%%%%%%% #
#     GPU 0       #
# %%%%%%%%%%%%%%% #

export CUDA_VISIBLE_DEVICES=0

# Experiment 1
OPTIM=adam
LR=0.001
NAME=${OPTIM}_lr${LR}_batch_size32  # name of experiment
mkdir -p $OUTDIR/$NAME
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
    > $OUTDIR/$NAME/terminal.txt &

# Experiment 2
OPTIM=adam
LR=0.001
NAME=${OPTIM}_lr${LR}_batch_size32_use_glove  # name of experiment
mkdir -p $OUTDIR/$NAME
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
    --use-glove \
    --glove-dir $TMP/glove \
    --optimizer $OPTIM \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    > $OUTDIR/$NAME/terminal.txt &

# Experiment 3
OPTIM=adam
LR=0.0005
NAME=${OPTIM}_lr${LR}_batch_size32  # name of experiment
mkdir -p $OUTDIR/$NAME
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
    > $OUTDIR/$NAME/terminal.txt &

# Experiment 4
OPTIM=adam
LR=0.0005
NAME=${OPTIM}_lr${LR}_batch_size32_use_glove  # name of experiment
mkdir -p $OUTDIR/$NAME
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
    --use-glove \
    --glove-dir $TMP/glove \
    --optimizer $OPTIM \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    > $OUTDIR/$NAME/terminal.txt &


# %%%%%%%%%%%%%%% #
#     GPU 1       #
# %%%%%%%%%%%%%%% #

export CUDA_VISIBLE_DEVICES=1

# Experiment 5
OPTIM=adam
LR=0.0001
NAME=${OPTIM}_lr${LR}_batch_size32  # name of experiment
mkdir -p $OUTDIR/$NAME
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
    > $OUTDIR/$NAME/terminal.txt &

# Experiment 6
OPTIM=adam
LR=0.0001
NAME=${OPTIM}_lr${LR}_batch_size32_use_glove  # name of experiment
mkdir -p $OUTDIR/$NAME
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
    --use-glove \
    --glove-dir $TMP/glove \
    --optimizer $OPTIM \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    > $OUTDIR/$NAME/terminal.txt &

# Experiment 7
OPTIM=rmsprop
LR=0.001
NAME=${OPTIM}_lr${LR}_batch_size32  # name of experiment
mkdir -p $OUTDIR/$NAME
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
    > $OUTDIR/$NAME/terminal.txt &

# Experiment 8
OPTIM=rmsprop
LR=0.001
NAME=${OPTIM}_lr${LR}_batch_size32_use_glove  # name of experiment
mkdir -p $OUTDIR/$NAME
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
    --use-glove \
    --glove-dir $TMP/glove \
    --optimizer $OPTIM \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    > $OUTDIR/$NAME/terminal.txt &


# %%%%%%%%%%%%%%% #
#     GPU 2       #
# %%%%%%%%%%%%%%% #

export CUDA_VISIBLE_DEVICES=2

# Experiment 9
OPTIM=sgd
LR=0.1
NAME=${OPTIM}_lr${LR}_batch_size32  # name of experiment
mkdir -p $OUTDIR/$NAME
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
    > $OUTDIR/$NAME/terminal.txt &

# Experiment 10
OPTIM=sgd
LR=0.1
NAME=${OPTIM}_lr${LR}_batch_size32_use_glove  # name of experiment
mkdir -p $OUTDIR/$NAME
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
    --use-glove \
    --glove-dir $TMP/glove \
    --optimizer $OPTIM \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    > $OUTDIR/$NAME/terminal.txt &

# Experiment 11
OPTIM=sgd
LR=0.05
NAME=${OPTIM}_lr${LR}_batch_size32  # name of experiment
mkdir -p $OUTDIR/$NAME
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
    > $OUTDIR/$NAME/terminal.txt &

# Experiment 12
OPTIM=sgd
LR=0.05
NAME=${OPTIM}_lr${LR}_batch_size32_use_glove  # name of experiment
mkdir -p $OUTDIR/$NAME
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
    --use-glove \
    --glove-dir $TMP/glove \
    --optimizer $OPTIM \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    > $OUTDIR/$NAME/terminal.txt &


# %%%%%%%%%%%%%%% #
#     GPU 3       #
# %%%%%%%%%%%%%%% #

export CUDA_VISIBLE_DEVICES=3

# Experiment 13
OPTIM=sgd
LR=0.01
NAME=${OPTIM}_lr${LR}_batch_size32  # name of experiment
mkdir -p $OUTDIR/$NAME
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
    > $OUTDIR/$NAME/terminal.txt &

# Experiment 14
OPTIM=sgd
LR=0.01
NAME=${OPTIM}_lr${LR}_batch_size32_use_glove  # name of experiment
mkdir -p $OUTDIR/$NAME
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
    --use-glove \
    --glove-dir $TMP/glove \
    --optimizer $OPTIM \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    > $OUTDIR/$NAME/terminal.txt &

# Experiment 15
OPTIM=sgd
LR=0.005
NAME=${OPTIM}_lr${LR}_batch_size32  # name of experiment
mkdir -p $OUTDIR/$NAME
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
    > $OUTDIR/$NAME/terminal.txt &

# Experiment 16
OPTIM=sgd
LR=0.005
NAME=${OPTIM}_lr${LR}_batch_size32_use_glove  # name of experiment
mkdir -p $OUTDIR/$NAME
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
    --use-glove \
    --glove-dir $TMP/glove \
    --optimizer $OPTIM \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    > $OUTDIR/$NAME/terminal.txt &

wait  # Wait for everyone to finish.


echo 'Finished training. Copying files from scratch...'
# Copy output directory from scratch to home
mkdir -p $EXP_DIR/$EXP_NAME
cp -r $OUTDIR/* $EXP_DIR/$EXP_NAME

# Cleanup scracth.
rm -r $TMP/*
