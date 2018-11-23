#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=90:00:00

# Loading modules
module load eb
module load Python/3.6.3-foss-2017b

# Home folders
SRCDIR=$HOME/thesis/dynet-rnng
DATADIR=$HOME/thesis/data
GLOVEDIR=$HOME/embeddings/glove
EXP_DIR=$SRCDIR/experiments/lisa
EXP_NAME=gen-job-cpu-5

# Scratch folders
TMP=$TMPDIR/daandir
OUTDIR=$TMP/results

# General training settings
MAX_TIME=$((80 * 3600))  # in seconds (related to -lwalltime)
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
ls -l $TMP/data

export MKL_NUM_THREADS=1

TRAIN=$TMP/data/train/ptb.train.trees
DEV=$TMP/data/dev/ptb.dev.trees
TEST=$TMP/data/test/ptb.test.trees
DEV_SAMPLES=$TMP/data/proposal-samples/dev.props
TEST_SAMPLES=$TMP/data/proposal-samples/test.props

OPTIM=sgd
LR=0.1
BATCH_SIZE=1

# Name of experiment.
for i in 1 2 3 4 5 6 7
do
    NAME=${OPTIM}_lr${LR}_batch_size${BATCH_SIZE}_use_glove_seed${i}
    mkdir -p $OUTDIR/$NAME

    python $SRCDIR/main.py train gen \
        --train-path $TRAIN \
        --dev-path $DEV \
        --test-path $TEST \
        --dev-proposal-samples $DEV_SAMPLES \
        --test-proposal-samples $TEST_SAMPLES \
        --use-glove \
        --glove-dir $TMP/glove \
        --action-emb-dim 100 \
        --stack-lstm-dim 256 \
        --terminal-lstm-dim 256 \
        --history-lstm-dim 256 \
        --dropout 0.3 \
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
        --lr-decay-patience 3 \
        --batch-size $BATCH_SIZE \
        --dynet-mem 2000 \
        --dynet-seed $((i)) \
        --seed $((i)) \
        > $OUTDIR/$NAME/terminal.txt &
done

# Name of experiment.
for i in 1 2 3 4 5 6 7
do
    NAME=${OPTIM}_lr${LR}_batch_size${BATCH_SIZE}_seed${i}
    mkdir -p $OUTDIR/$NAME

    python $SRCDIR/main.py train gen \
        --train-path $TRAIN \
        --dev-path $DEV \
        --test-path $TEST \
        --dev-proposal-samples $DEV_SAMPLES \
        --test-proposal-samples $TEST_SAMPLES \
        --action-emb-dim 100 \
        --stack-lstm-dim 256 \
        --terminal-lstm-dim 256 \
        --history-lstm-dim 256 \
        --dropout 0.3 \
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
        --lr-decay-patience 3 \
        --batch-size $BATCH_SIZE \
        --dynet-mem 2000 \
        --dynet-seed $((i)) \
        --seed $((i)) \
        > $OUTDIR/$NAME/terminal.txt &
done

wait
echo 'Finished training. Copying files from scratch...'
# Copy output directory from scratch to home
mkdir -p $EXP_DIR/$EXP_NAME
cp -r $OUTDIR/* $EXP_DIR/$EXP_NAME

# Cleanup scracth.
rm -r $TMP/*
