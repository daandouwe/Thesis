# picks model with highest fscore or lowest perplexity (for evaluation and semisup training)
CRF_PATH ?= models/$(ls models | grep '^crf_dev=[0-9.]\+$' | cut -c9- | sort -n | tail -n1 | sed 's|^|crf_dev=|')
DISC_PATH ?= models/$(ls models | grep '^disc-rnng_dev=[0-9.]\+$' | cut -c15- | sort -n | tail -n1 | sed 's|^|disc-rnng_dev=|')
GEN_PATH ?= models/$(ls models | grep '^gen-rnng_dev=[0-9.]\+$' | cut -c14- | sort -n | tail -n1 | sed 's|^|gen-rnng_dev=|')

CRF_VOCAB_PATH ?= models/$(ls models | grep '^crf_vocab=semisup_dev=[0-9.]\+$' | cut -c23- | sort -n | tail -n1 | sed 's|^|crf_vocab=semisup_dev=|')
DISC_VOCAB_PATH ?= models/$(ls models | grep '^disc-rnng_vocab=semisup_dev=[0-9.]\+$' | cut -c29- | sort -n | tail -n1 | sed 's|^|disc-rnng_vocab=semisup_dev=|')
GEN_VOCAB_PATH ?= models/$(ls models | grep '^gen-rnng_vocab=semisup_dev=[0-9.]\+$' | cut -c28- | sort -n | tail -n1 | sed 's|^|gen-rnng_vocab=semisup_dev=|')

LM_PATH ?= models/$(ls models | grep '^lm_dev=[0-9.]\+$' | cut -c8- | sort -nr | tail -n1 | sed 's|^|lm_dev=|')
MULTI_LM_PATH ?= models/$(ls models | grep '^multitask-lm_dev=[0-9.]\+$' | cut -c18- | sort -nr | tail -n1 | sed 's|^|multitask-lm_dev=|')


# setup
evalb:
	scripts/install-evalb.sh

data:
	mkdir data && scripts/get-ptb.sh && scripts/get-unlabeled.sh  && scripts/get-syneval.sh

# training
disc: train-disc
gen: train-gen
crf: train-crf
lm: train-lm
multitask-lm: train-multitask-lm

disc-vocab: semisup-vocab train-disc-vocab
gen-vocab: semisup-vocab train-gen-vocab
crf-vocab: semisup-vocab train-crf-vocab

disc-lowercase-vocab: semisup-lowercase-vocab train-disc-vocab
gen-lowercase-vocab: semisup-lowercase-vocab train-gen-vocab
crf-lowercase-vocab: semisup-lowercase-vocab train-crf-vocab

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
	    --model-path-base=models/disc-rnng \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/disc-rnng.txt \
	    @src/configs/training/sgd.txt

train-gen:
	python src/main.py train \
	    --model-path-base=models/gen-rnng \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/gen-rnng.txt \
	    @src/configs/training/sgd.txt \
	    @src/configs/proposals/rnng.txt \

train-crf:
	python src/main.py train \
	    --model-path-base=models/crf \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/crf.txt \
	    @src/configs/training/adam.txt \

train-lm:
	python src/main.py train \
	    --model-path-base=models/lm \
			@src/configs/data/supervised.txt \
			@src/configs/model/lm.txt \
			@src/configs/training/sgd.txt \
			--batch-size=64

train-multitask-lm:
	python src/main.py train \
	    --model-path-base=models/multitask-lm \
			@src/configs/data/supervised.txt \
			@src/configs/model/multitask-lm.txt \
			@src/configs/training/sgd.txt \
			--batch-size=64

train-disc-vocab: semisup-vocab
	python src/main.py train \
	    --model-path-base=models/disc-rnng_vocab=semisup \
	    @src/configs/vocab/semisupervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/disc-rnng.txt \
	    @src/configs/training/sgd.txt

train-gen-vocab: semisup-vocab
	python src/main.py train \
	    --model-path-base=models/gen-rnng_vocab=semisup \
	    @src/configs/vocab/semisupervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/gen-rnng.txt \
	    @src/configs/training/sgd.txt \
	    @src/configs/proposal/disc-rnng.txt

train-crf-vocab: semisup-vocab
	python src/main.py train \
	    --model-path-base=models/crf_vocab=semisup \
	    @src/configs/vocab/semisupervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/crf.txt \
	    @src/configs/training/adam.txt

# finetune models semisupervised
semisup-rnng:
	python src/main.py semisup \
	    --model-path-base=models/semisup_post=disc \
			--joint-model=${GEN_PATH}/model \
			--posterior-model=${DISC_PATH}/model \
	    @src/configs/data/semisupervised.txt \
	    @src/configs/training/adam.txt \
	    @src/configs/baseline/argmax.txt

semisup-crf:
	python src/main.py semisup \
	    --model-path-base=models/semisup_post=crf \
			--joint-model=${GEN_PATH}/model \
			--posterior-model=${CRF_PATH}/model \
	    @src/configs/data/semisupervised.txt \
	    @src/configs/training/adam.txt \
	    @src/configs/baseline/argmax.txt

semisup-rnng-vocab:
	python src/main.py semisup \
	    --model-path-base=models/semisup_post=disc_vocab=semisup \
			--joint-model=${GEN_VOCAB_PATH}/model \
			--posterior-model=${DISC_VOCAB_PATH}/model \
	    @src/configs/vocab/semisupervised.txt \
	    @src/configs/data/semisupervised.txt \
	    @src/configs/training/adam.txt \
	    @src/configs/baseline/argmax.txt

semisup-crf-vocab:
	python src/main.py semisup \
	    --model-path-base=models/semisup_post=crf_vocab=semisup \
			--joint-model=${GEN_VOCAB_PATH}/model \
			--posterior-model=${CRF_VOCAB_PATH}/model \
	    @src/configs/vocab/semisupervised.txt \
	    @src/configs/data/semisupervised.txt \
	    @src/configs/training/adam.txt \
	    @src/configs/baseline/argmax.txt

# sample proposals
proposals-rnng: proposals-rnng-dev proposals-rnng-test
proposals-crf: proposals-crf-dev proposals-crf-test

proposals-rnng-dev:
	python src/main.py predict \
	    @src/configs/proposals/sample-rnng-dev.txt

proposals-rnng-test:
	python src/main.py predict \
	    @src/configs/proposals/sample-rnng-test.txt

proposals-crf-dev:
	python src/main.py predict \
	    @src/configs/proposals/sample-crf-dev.txt

proposals-crf-test:
	python src/main.py predict \
	    @src/configs/proposals/sample-crf-test.txt

# eval
eval-pp:
	python src/main.py predict

# WTF this doesn't work, LM_PATH is empty
syneval-lm:
	python src/main.py syneval \
	    --parser-type=rnn-lm \
	    --checkpoint=${LM_PATH}/model \
	    --model-path-base= \
	    --indir=data/syneval/converted
			--add-period \
			--capitalize

.PHONY : clean
clean :
	-rm -r models/temp/*
