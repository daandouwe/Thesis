PTB ?= data/ptb/02-21.10way.clean
CCG ?= data/ccg/train.text
SYN ?= data/syneval/data/converted
OBW ?= news.en-00001-of-00100.processed

GEN_EVAL_EVERY ?= 4
DEV_NUM_SAMPLES ?= 50
TEST_NUM_SAMPLES ?= 100

SEMISUP_NUM_SAMPLES ?= 8
SEMISUP_BATCH_SIZE ?= 1

UNSUP_NUM_SAMPLES ?= 8
UNSUP_BATCH_SIZE ?= 1

SYN_NUM_SAMPLES ?= 50
SYN_MAX_LINES ?= 1000

ALPHA ?= 0.8  # temperature for samples (discriminative rnng)
SEED ?= 42

# Note: use `source scripts/best-models.sh` to set variable names to paths
# of the best development models, e.g. export GEN_PATH=models/gen-rnng_dev=100.93


# list all available actions
list :
	grep -o '^.*[^ ]:' Makefile | rev | cut '-c2-' | rev | sort -u

# clear temporary models
clean :
	-rm -r models/temp/*


# setup and data
evalb:
	scripts/install-evalb.sh

data:
	mkdir -p data && scripts/get-ptb.sh && scripts/get-unlabeled.sh && scripts/get-syneval.sh

silver: predict-silver
	cat data/silver/${OBW}.trees ${PTB} | gshuf > data/silver/silver-gold.trees


# train a model
disc: train-disc
crf: train-crf

gen: vocab-sup train-gen
gen-stack-only: vocab-sup train-gen-stack-only
gen-gpu: vocab-sup train-gen-gpu
gen-silver: vocab-semisup train-gen-silver

lm: vocab-sup train-lm
lm-gpu: vocab-sup train-lm-gpu
lm-multitask-span: vocab-sup train-lm-multitask-span
lm-multitask-span-gpu: vocab-sup train-lm-multitask-span-gpu
lm-multitask-ccg: vocab-sup train-lm-multitask-ccg
lm-multitask-ccg-gpu: vocab-sup train-lm-multitask-ccg-gpu

disc-semisup-vocab: vocab-semisup train-disc-vocab
crf-semisup-vocab: vocab-semisup train-crf-vocab
gen-semisup-vocab: vocab-semisup train-gen-vocab

proposals-rnng: proposals-rnng-dev proposals-rnng-test
proposals-crf: proposals-crf-dev proposals-crf-test

semisup-disc: train-semisup-disc
semisup-crf: train-semisup-crf

fully-unsup-disc: vocab-sup train-fully-unsup-disc
fully-unsup-crf: vocab-sup train-fully-unsup-crf


# build a vocabulary
vocab-sup:
	python src/main.py build \
	    @src/configs/vocab/supervised.txt \

vocab-semisup:
	python src/main.py build \
	    @src/configs/vocab/semisupervised.txt


# train a supervised model
train-disc:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --model-path-base=models/disc-rnng \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/disc-rnng.txt \
	    @src/configs/training/sgd.txt \
	    --print-every 1

train-gen:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/gen-rnng \
	    @src/configs/vocab/supervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/gen-rnng.txt \
	    @src/configs/training/sgd.txt \
	    @src/configs/proposals/rnng.txt \
	    --eval-every-epochs=${GEN_EVAL_EVERY} \
	    --num-dev-samples=${DEV_NUM_SAMPLES} \
	    --num-test-samples=${TEST_NUM_SAMPLES} \
	    --unlabeled \
	    --print-every 1

train-gen-stack-only:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/gen-rnng_stack-only \
	    @src/configs/vocab/supervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/gen-rnng-stack-only.txt \
	    @src/configs/training/sgd.txt \
	    @src/configs/proposals/rnng.txt \
	    --eval-every-epochs=${GEN_EVAL_EVERY} \
	    --num-dev-samples=${DEV_NUM_SAMPLES} \
	    --num-test-samples=${TEST_NUM_SAMPLES}

train-gen-silver:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/gen-rnng_data=silver-gold \
	    @src/configs/vocab/semisupervised.txt \
	    @src/configs/data/supervised-silver.txt \
	    @src/configs/model/gen-rnng.txt \
	    @src/configs/training/sgd.txt \
	    @src/configs/proposals/rnng.txt \
	    --eval-every-epochs=${GEN_EVAL_EVERY} \
	    --num-dev-samples=${DEV_NUM_SAMPLES} \
	    --num-test-samples=${TEST_NUM_SAMPLES}

train-crf:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/crf \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/crf.txt \
	    @src/configs/training/sgd.txt \
	    --min-label-count 100 \
	    --print-every=10

train-lm:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/lm \
	    @src/configs/vocab/supervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/lm.txt \
	    @src/configs/training/sgd.txt

train-lm-multitask-span:
	python src/main.py train \
      --dynet-autobatch=1 \
	    --dynet-mem=1000 \
	    --model-path-base=models/lm-multitask-span \
	    @src/configs/vocab/supervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/lm-multitask-span.txt \
	    @src/configs/training/sgd.txt

train-lm-multitask-ccg:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=1000 \
	    --model-path-base=models/lm-multitask-ccg \
	    @src/configs/vocab/supervised.txt \
	    @src/configs/data/supervised-ccg.txt \
	    @src/configs/model/lm-multitask-ccg.txt \
	    @src/configs/training/sgd.txt

# gpu models
train-gen-gpu:
	python src/main.py train \
	    --dynet-gpus=1 \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/gen-rnng \
	    @src/configs/vocab/supervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/gen-rnng.txt \
	    @src/configs/training/sgd.txt \
	    @src/configs/proposals/rnng.txt \
	    --eval-every-epochs=${GEN_EVAL_EVERY} \
	    --num-dev-samples=${DEV_NUM_SAMPLES} \
	    --num-test-samples=${TEST_NUM_SAMPLES}

train-lm-gpu:
	python src/main.py train \
	    --dynet-gpus=1 \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/lm \
	    @src/configs/vocab/supervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/lm.txt \
	    @src/configs/training/sgd.txt

train-lm-multitask-span-gpu:
	python src/main.py train \
	    --dynet-gpus=1 \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/multitask-span-lm \
	    @src/configs/vocab/supervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/lm-multitask-span.txt \
	    @src/configs/training/sgd.txt

train-lm-multitask-ccg-gpu:
	python src/main.py train \
	    --dynet-gpus=1 \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/multitask-ccg-lm \
	    @src/configs/vocab/supervised.txt \
	    @src/configs/data/supervised-ccg.txt \
	    @src/configs/model/lm-multitask-ccg.txt \
	    @src/configs/training/sgd.txt


# semisup vocab models
train-disc-vocab: vocab-semisup
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=1000 \
	    --model-path-base=models/disc-rnng_vocab=semisup \
	    @src/configs/vocab/semisupervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/disc-rnng.txt \
	    @src/configs/training/sgd.txt

train-gen-vocab: vocab-semisup
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/gen-rnng_vocab=semisup \
	    @src/configs/vocab/semisupervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/gen-rnng.txt \
	    @src/configs/training/sgd.txt \
	    @src/configs/proposals/rnng.txt \
	    --eval-every-epochs=${GEN_EVAL_EVERY} \
	    --num-dev-samples=${DEV_NUM_SAMPLES} \
	    --num-test-samples=${TEST_NUM_SAMPLES}

train-crf-vocab: vocab-semisup
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=1000 \
	    --model-path-base=models/crf_vocab=semisup \
	    @src/configs/vocab/semisupervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/crf.txt \
	    @src/configs/training/adam.txt


# semisupervised training from pretrained models
train-semisup-disc:
	python src/main.py train \
    	--dynet-autobatch=1 \
	    --dynet-mem=6000 \
	    --model-path-base=models/semisup-disc \
	    --model-type=semisup-disc \
	    --joint-model-path=${GEN_PATH} \
	    --post-model-path=${DISC_PATH} \
	    @src/configs/vocab/semisupervised.txt \
	    @src/configs/data/semisupervised.txt \
	    @src/configs/training/adam.txt \
	    --use-argmax-baseline \
	    --num-samples=${SEMISUP_NUM_SAMPLES} \
	    --batch-size=${SEMISUP_BATCH_SIZE} \
	    --num-dev-samples=${DEV_NUM_SAMPLES} \
	    --num-test-samples=${TEST_NUM_SAMPLES} \
	    --print-every=1

train-semisup-crf:
	python src/main.py train \
    	--dynet-autobatch=1 \
	    --dynet-mem=6000 \
	    --model-path-base=models/semisup-crf \
	    --model-type=semisup-crf \
	    --joint-model-path=${GEN_PATH} \
	    --post-model-path=${CRF_PATH} \
	    @src/configs/vocab/semisupervised.txt \
	    @src/configs/data/semisupervised.txt \
	    @src/configs/training/adam.txt \
	    --use-argmax-baseline \
	    --num-samples=${SEMISUP_NUM_SAMPLES} \
	    --batch-size=${SEMISUP_BATCH_SIZE} \
	    --num-dev-samples=${DEV_NUM_SAMPLES} \
	    --num-test-samples=${TEST_NUM_SAMPLES} \
	    --print-every=1

train-unsup-disc:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=6000 \
	    --model-path-base=models/unsup-disc \
	    --model-type=unsup-disc \
	    --joint-model-path=${GEN_PATH} \
	    --post-model-path=${DISC_PATH} \
	    @src/configs/vocab/supervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/training/adam.txt \
	    --use-argmax-baseline \
	    --num-samples=${UNSUP_NUM_SAMPLES} \
	    --batch-size=${UNSUP_BATCH_SIZE} \
	    --num-dev-samples=${DEV_NUM_SAMPLES} \
	    --num-test-samples=${TEST_NUM_SAMPLES} \
	    --print-every=1

train-unsup-crf:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=6000 \
	    --model-path-base=models/unsup-crf \
	    --model-type=unsup-crf \
	    --joint-model-path=${GEN_PATH} \
	    --post-model-path=${CRF_PATH} \
	    @src/configs/vocab/supervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/training/adam.txt \
	    --use-argmax-baseline \
	    --num-samples=${UNSUP_NUM_SAMPLES} \
	    --batch-size=${UNSUP_BATCH_SIZE} \
	    --num-dev-samples=${DEV_NUM_SAMPLES} \
	    --num-test-samples=${TEST_NUM_SAMPLES} \
	    --exact-entropy \
	    --print-every=1

# semisupervised and unsupervised without pretraining
train-fully-unsup-crf:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=9000 \
	    --model-path-base=models/fully-unsup-crf \
	    --model-type=unsup-crf \
	    @src/configs/vocab/supervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/training/adam.txt \
	    --unlabeled \
	    --num-samples=${UNSUP_NUM_SAMPLES} \
	    --batch-size=${UNSUP_BATCH_SIZE} \
	    --num-dev-samples=${DEV_NUM_SAMPLES} \
	    --num-test-samples=${TEST_NUM_SAMPLES} \
	    --print-every=1 \
	    --anneal-entropy \
	    --num-anneal-epochs 1


train-fully-unsup-disc:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=6000 \
	    --model-path-base=models/fully-unsup-disc \
	    --model-type=unsup-disc \
	    @src/configs/vocab/supervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/training/adam.txt \
	    --use-argmax-baseline \
	    --unlabeled \
	    --num-samples=${UNSUP_NUM_SAMPLES} \
	    --batch-size=${UNSUP_BATCH_SIZE} \
	    --num-dev-samples=${DEV_NUM_SAMPLES} \
	    --num-test-samples=${TEST_NUM_SAMPLES} \
	    --print-every=1

train-fully-semisup-crf:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=9000 \
	    --model-path-base=models/fully-semisup-crf \
	    --model-type=semisup-crf \
	    @src/configs/vocab/supervised.txt \
	    @src/configs/data/semisupervised.txt \
	    --unlabeled-path=${PTB} \
	    @src/configs/training/adam.txt \
	    --unlabeled \
	    --num-samples=${UNSUP_NUM_SAMPLES} \
	    --batch-size=${UNSUP_BATCH_SIZE} \
	    --num-dev-samples=${DEV_NUM_SAMPLES} \
	    --num-test-samples=${TEST_NUM_SAMPLES} \
	    --print-every=1


# resume training a model
resume-train:
	# get rid of eplicit switches, e.g. `--lowercase=False`, and replace `_` by `-`.
	cat "${RESUME_PATH}/log/args.txt" | grep -v 'False' > 'args_.txt' | mv 'args_.txt' "${RESUME_PATH}/log/args.txt" | sed -i '' 's/_/-/g' "${RESUME_PATH}/log/args.txt"
	python src/main.py train \
	    @${RESUME_PATH}/log/args.txt \
	    --resume=${RESUME_PATH}


# sample proposals
proposals-rnng-dev:
	python src/main.py predict \
	    --checkpoint=${DISC_PATH} \
	    @src/configs/proposals/sample-rnng-dev.txt

proposals-rnng-test:
	python src/main.py predict \
	    --checkpoint=${DISC_PATH} \
	    @src/configs/proposals/sample-rnng-test.txt

proposals-crf-dev:
	python src/main.py predict \
	    --checkpoint=${CRF_PATH} \
	    @src/configs/proposals/sample-crf-dev.txt

proposals-crf-test:
	python src/main.py predict \
	    --checkpoint=${CRF_PATH} \
	    @src/configs/proposals/sample-crf-test.txt


# predict silver training trees
predict-silver:
	mkdir -p data/silver
	python src/main.py predict \
	    --from-text-file \
	    --model-type=disc-rnng \
	    --checkpoint=${DISC_PATH} \
	    --infile=data/unlabeled/${OBW} \
	    --outfile=data/silver/${OBW}.trees


# predict from terminal input
predict-input-disc:
	python src/main.py predict \
	    --from-input \
	    --model-type=disc-rnng \
	    --checkpoint=${DISC_PATH} \
	    --alpha=${ALPHA}

predict-input-crf:
	python src/main.py predict \
	    --from-input \
	    --model-type=crf \
	    --checkpoint=${CRF_PATH}

predict-input-gen:
	python src/main.py predict \
	    --from-input \
	    --model-type=gen-rnng \
	    --checkpoint=${GEN_PATH} \
	    --proposal-model=${DISC_PATH} \
	    --alpha=${ALPHA}

predict-input-gen-crf:
	python src/main.py predict \
	    --from-input \
	    --model-type=gen-rnng \
	    --checkpoint=${GEN_PATH} \
	    --proposal-model=${CRF_PATH}


# compute entropies
predict-entropy-crf:
	python src/main.py predict \
	    --entropy \
	    --model-type=crf \
	    --infile=${INFILE} \
	    --outfile=${OUTFILE} \
	    --checkpoint=${CRF_PATH} \
	    --num-samples=${NUM_SAMPLES}


# evaluate perplexity
perplexity-test:
	python src/main.py predict \
		  --dynet-autobatch=1 \
		  --dynet-mem=2000 \
		  --model-type=gen-rnng \
		  --perplexity \
		  --checkpoint=${GEN_PATH} \
		  --proposal-samples=${PROP_PATH} \
		  --num-samples=${TEST_NUM_SAMPLES} \
		  --outfile=${GEN_PATH}/output/results.tsv


# syneval
syneval-lm:
	python src/main.py syneval \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-type=rnn-lm \
	    --checkpoint=${LM_PATH} \
	    --indir=${SYN}

syneval-multitask-span-lm:
	python src/main.py syneval \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-type=rnn-lm \
	    --checkpoint=${LM_MULTI_CCG_PATH} \
	    --indir=${SYN}

syneval-multitask-ccg-lm:
	python src/main.py syneval \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-type=rnn-lm \
	    --checkpoint=${LM_MULTI_CCG_PATH} \
	    --indir=${SYN}

syneval-rnng:
	python src/main.py syneval \
	    --dynet-autobatch=1 \
	    --dynet-mem=5000 \
	    --model-type=gen-rnng \
	    --checkpoint=${GEN_PATH} \
	    --proposal-model=${DISC_PATH} \
	    --indir=${SYN} \
	    --num-samples=${SYN_NUM_SAMPLES} \
	    --syneval-max-lines=${SYN_MAX_LINES} \

syneval-rnng-crf:
	python src/main.py syneval \
	    --dynet-autobatch=1 \
	    --dynet-mem=5000 \
	    --model-type=gen-rnng \
	    --checkpoint=${GEN_PATH} \
	    --proposal-model=${CRF_PATH} \
	    --indir=${SYN} \
	    --alpha=1.0 \
	    --num-samples=${SYN_NUM_SAMPLES} \
	    --syneval-max-lines=${SYN_MAX_LINES} \


# experimental: syneval classification with entropy
syneval-disc:
	python src/main.py syneval \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-type=disc-rnng \
	    --checkpoint=${DISC_PATH} \
	    --indir=${SYN} \
	    --num-samples=${SYN_NUM_SAMPLES}

syneval-crf:
	python src/main.py syneval \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-type=disc-rnng \
	    --checkpoint=${CRF_PATH} \
	    --indir=${SYN} \
	    --num-samples=0 \
	    --exact-entropy \
	    --syneval-short
