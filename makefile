
# setup
evalb:
	scripts/install-evalb.sh

data:
	mkdir data && scripts/get-ptb.sh && scripts/get-unlabeled.sh  && scripts/get-syneval.sh

# train a model
disc: train-disc
gen: train-gen
crf: train-crf
lm: train-lm
multitask-lm: train-multitask-lm

gen-gpu: train-gen-gpu
lm-gpu: train-lm-gpu
multitask-lm-gpu: train-multitask-lm-gpu

disc-vocab: semisup-vocab train-disc-vocab
gen-vocab: semisup-vocab train-gen-vocab
crf-vocab: semisup-vocab train-crf-vocab

disc-lowercase-vocab: semisup-lowercase-vocab train-disc-vocab
gen-lowercase-vocab: semisup-lowercase-vocab train-gen-vocab
crf-lowercase-vocab: semisup-lowercase-vocab train-crf-vocab

gen-stack-only: train-gen-stack-only

semisup-rnng: train-semisup-rnng
semisup-crf: train-semisup-crf
semisup-rnng-vocab: train-semisup-rnng-vocab
semisup-crf-vocab: train-semisup-crf-vocab


# build a vocabulary
sup-vocab:
	python src/build.py \
	    @src/configs/vocab/supervised.txt

sup-lowercase-vocab:
	python src/build.py \
	    @src/configs/vocab/supervised.txt \
	    --lowercase

semisup-vocab:
	python src/build.py \
	    @src/configs/vocab/semisupervised.txt

semisup-lowercase-vocab:
	python src/build.py \
	    @src/configs/vocab/semisupervised.txt \
	    --lowercase

# Train a supervised model
train-disc:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --model-path-base=models/disc-rnng \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/disc-rnng.txt \
	    @src/configs/training/sgd.txt

train-gen:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/gen-rnng \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/gen-rnng.txt \
	    @src/configs/training/sgd.txt \
	    @src/configs/proposals/rnng.txt

train-crf:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=1000 \
	    --model-path-base=models/crf \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/crf.txt \
	    @src/configs/training/adam.txt \

train-lm:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=1000 \
	    --model-path-base=models/lm \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/lm.txt \
	    @src/configs/training/sgd.txt

train-multitask-lm:
	python src/main.py train \
      --dynet-autobatch=1 \
	    --dynet-mem=1000 \
	    --model-path-base=models/multitask-lm \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/multitask-lm.txt \
	    @src/configs/training/sgd.txt

train-gen-gpu:
	python src/main.py train \
	    --dynet-gpus=1 \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/gen-rnng \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/gen-rnng.txt \
	    @src/configs/training/sgd.txt \
	    @src/configs/proposals/rnng.txt

train-lm-gpu:
	python src/main.py train \
	    --dynet-gpus=1 \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/lm \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/lm.txt \
	    @src/configs/training/sgd.txt

train-multitask-lm-gpu:
	python src/main.py train \
	    --dynet-gpus=1 \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/multitask-lm \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/multitask-lm.txt \
	    @src/configs/training/sgd.txt

train-disc-vocab: semisup-vocab
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=1000 \
	    --model-path-base=models/disc-rnng_vocab=semisup \
	    @src/configs/vocab/semisupervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/disc-rnng.txt \
	    @src/configs/training/sgd.txt

train-gen-vocab: semisup-vocab
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/gen-rnng_vocab=semisup \
	    @src/configs/vocab/semisupervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/gen-rnng.txt \
	    @src/configs/training/sgd.txt \
	    @src/configs/proposal/disc-rnng.txt

train-crf-vocab: semisup-vocab
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=1000 \
	    --model-path-base=models/crf_vocab=semisup \
	    @src/configs/vocab/semisupervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/crf.txt \
	    @src/configs/training/adam.txt

train-gen-stack-only:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/gen-rnng_stack-only \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/gen-rnng-stack-only.txt \
	    @src/configs/training/sgd.txt \
	    @src/configs/proposals/rnng.txt


# finetune models semisupervised
train-semisup-rnng:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=6000 \
	    --model-path-base=models/semisup_post=disc \
	    --model-type=semisup-rnng \
	    --joint-model-path=${GEN_PATH} \
	    --post-model-path=${DISC_PATH} \
	    --num-samples=1 \
	    @src/configs/data/semisupervised.txt \
	    @src/configs/training/adam.txt \
	    @src/configs/baseline/argmax.txt \
	    --batch-size=1

train-semisup-crf:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=6000 \
	    --model-path-base=models/semisup_post=crf \
	    --model-type=semisup-crf \
	    --joint-model-path=${GEN_PATH} \
	    --post-model-path=${CRF_PATH} \
	    --num-samples=1 \
	    @src/configs/data/semisupervised.txt \
	    @src/configs/training/adam.txt \
	    @src/configs/baseline/argmax.txt \
	    --batch-size=1 \

train-semisup-rnng-vocab:
	python src/main.py train \
    	--dynet-autobatch=1 \
	    --dynet-mem=6000 \
	    --model-path-base=models/semisup_post=disc_vocab=semisup \
	    --model-type=semisup-rnng \
	    --joint-model-path=${GEN_VOCAB_PATH} \
	    --post-model-path=${DISC_VOCAB_PATH} \
	    --num-samples=1 \
	    @src/configs/vocab/semisupervised.txt \
	    @src/configs/data/semisupervised.txt \
	    @src/configs/training/adam.txt \
	    @src/configs/baseline/argmax.txt \
	    --batch-size=1 \

train-semisup-crf-vocab:
	python src/main.py train \
    	--dynet-autobatch=1 \
	    --dynet-mem=6000 \
	    --model-path-base=models/semisup_post=crf_vocab=semisup \
	    --model-type=semisup-crf \
	    --joint-model-path=${GEN_VOCAB_PATH} \
	    --post-model-path=${CRF_VOCAB_PATH} \
	    --num-samples=1 \
	    @src/configs/vocab/semisupervised.txt \
	    @src/configs/data/semisupervised.txt \
	    @src/configs/training/adam.txt \
	    @src/configs/baseline/argmax.txt \
	    --batch-size=1 \

# sample proposals
proposals-rnng: proposals-rnng-dev proposals-rnng-test
proposals-crf: proposals-crf-dev proposals-crf-test

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

# evaluation
eval-test-pp:
	python src/main.py predict \
	    --dynet-autobatch=1 \
	    --dynet-mem=5000 \
	    --model-type=gen-rnng \
	    --model-path-base='' \
	    --perplexity \
	    --checkpoint=${GEN_PATH} \
	    --infile=data/ptb/test.trees \
	    --proposal-samples=data/proposals/rnng-test.props \
	    --num-samples=100

syneval-lm:
	python src/main.py syneval \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-type=rnn-lm \
	    --checkpoint=${LM_PATH} \
	    --model-path-base='' \
	    --indir=data/syneval/converted \
	    --capitalize

syneval-multitask-lm:
	python src/main.py syneval \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-type=rnn-lm \
	    --checkpoint=${MULTI_LM_PATH} \
	    --model-path-base='' \
	    --indir=data/syneval/converted \
	    --capitalize

syneval-rnng:
	python src/main.py syneval \
	    --dynet-autobatch=1 \
	    --dynet-mem=5000 \
	    --model-type=gen-rnng \
	    --checkpoint=${GEN_PATH} \
	    --proposal-model=${DISC_PATH} \
	    --model-path-base='' \
	    --indir=data/syneval/converted \
	    --num-samples=20 \
	    --capitalize

syneval-disc:
	python src/main.py syneval \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-type=disc-rnng \
	    --checkpoint=${DISC_PATH} \
	    --model-path-base='' \
	    --indir=data/syneval/converted \
	    --num-samples=10 \
	    --capitalize \
	    --add-period \
	    --syneval-short


# clear temporary models
.PHONY : clean
clean :
	-rm -r models/temp/*
