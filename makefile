PTB=data/ptb/train.trees
OBW=news.en-00001-of-00100.processed


# setup
evalb:
	scripts/install-evalb.sh

data:
	mkdir data && scripts/get-ptb.sh && scripts/process-unlabeled.sh && scripts/get-syneval.sh

silver: predict-silver
	cat data/silver/${OBW}.trees ${PTB} | gshuf > data/silver/silver-gold.trees


# train a model
disc: train-disc
gen: sup-vocab train-gen
crf: train-crf
lm: sup-vocab train-lm
multitask-lm: sup-vocab train-multitask-lm

gen-stack-only: sup-vocab train-gen-stack-only

gen-gpu: sup-vocab train-gen-gpu
lm-gpu: sup-vocab train-lm-gpu
multitask-lm-gpu: sup-vocab train-multitask-lm-gpu

disc-semisup-vocab: semisup-vocab train-disc-vocab
gen-semisup-vocab: semisup-vocab train-gen-vocab
crf-semisup-vocab: semisup-vocab train-crf-vocab

semisup-rnng: train-semisup-rnng
semisup-crf: train-semisup-crf
semisup-rnng-vocab: train-semisup-rnng-vocab
semisup-crf-vocab: train-semisup-crf-vocab

gen-silver: semisup-vocab train-gen-silver


# build a vocabulary
sup-vocab:
	python src/main.py build \
	    @src/configs/vocab/supervised.txt \

semisup-vocab:
	python src/main.py build \
	    @src/configs/vocab/semisupervised.txt


# train a supervised model
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
	    @src/configs/vocab/supervised.txt \
			@src/configs/data/supervised.txt \
	    @src/configs/model/gen-rnng.txt \
	    @src/configs/training/sgd.txt \
	    @src/configs/proposals/rnng.txt \
	    --eval-every-epochs=4 \
	    --num-dev-samples=50 \
	    --num-test-samples=100

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
	    --eval-every-epochs=4 \
	    --num-dev-samples=50 \
	    --num-test-samples=100

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
	    --eval-every-epochs=4 \
	    --num-dev-samples=50 \
	    --num-test-samples=100


train-crf:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/crf \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/crf.txt \
	    @src/configs/training/sgd.txt

train-lm:
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/lm \
			@src/configs/vocab/supervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/lm.txt \
	    @src/configs/training/sgd.txt

train-multitask-lm:
	python src/main.py train \
      --dynet-autobatch=1 \
	    --dynet-mem=1000 \
	    --model-path-base=models/multitask-lm \
			@src/configs/vocab/supervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/multitask-lm.txt \
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
	    --eval-every-epochs=4 \
	    --num-dev-samples=50 \
	      --num-test-samples=100

train-lm-gpu:
	python src/main.py train \
	    --dynet-gpu=1 \
			--dynet-gpus=1 \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/lm \
			@src/configs/vocab/supervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/lm.txt \
	    @src/configs/training/sgd.txt

train-multitask-lm-gpu:
	python src/main.py train \
	    --dynet-gpus=1 \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-path-base=models/multitask-lm \
			@src/configs/vocab/supervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/multitask-lm.txt \
	    @src/configs/training/sgd.txt


# semisup vocab models
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
	    @src/configs/proposals/rnng.txt \
	    --eval-every-epochs=4 \
	    --num-dev-samples=50 \
	    --num-test-samples=100

train-crf-vocab: semisup-vocab
	python src/main.py train \
	    --dynet-autobatch=1 \
	    --dynet-mem=1000 \
	    --model-path-base=models/crf_vocab=semisup \
	    @src/configs/vocab/semisupervised.txt \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/crf.txt \
	    @src/configs/training/adam.txt


# semisupervised training
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
	    @src/configs/baseline/mlp.txt \
	    --batch-size=1 \
	    --num-dev-samples=50 \
	    --num-test-samples=100

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
	    --num-dev-samples=50 \
	    --num-test-samples=100

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
	    --num-dev-samples=50 \
	    --num-test-samples=100

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
	    --num-dev-samples=50 \
	    --num-test-samples=100


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

# predict silver training trees
predict-silver:
	mkdir -p data/silver && \
	python src/main.py predict \
	    --from-text-file \
	    --model-type=disc-rnng \
	    --checkpoint=${DISC_PATH} \
	    --infile=data/unlabeled/${OBW} \
	    --outfile=data/silver/${OBW}.trees


# evaluate perplexity
eval-test-pp:
	python src/main.py predict \
	    --dynet-autobatch=1 \
	    --dynet-mem=5000 \
	    --model-type=gen-rnng \
	    --perplexity \
	    --checkpoint=${GEN_PATH} \
	    --infile=data/ptb/test.trees \
	    --proposal-samples=data/proposals/rnng-test.props \
	    --num-samples=100

# syneval
syneval-lm:
	python src/main.py syneval \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-type=rnn-lm \
	    --checkpoint=${LM_PATH} \
	    --indir=data/syneval/data/converted

syneval-multitask-lm:
	python src/main.py syneval \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-type=rnn-lm \
	    --checkpoint=${MULTI_LM_PATH} \
	    --indir=data/syneval/data/converted

syneval-rnng:
	python src/main.py syneval \
	    --dynet-autobatch=1 \
	    --dynet-mem=5000 \
	    --model-type=gen-rnng \
	    --checkpoint=${GEN_PATH} \
	    --proposal-model=${DISC_PATH} \
	    --indir=data/syneval/data/converted \
	    --num-samples=50 \
	    --syneval-max-lines=1000 \

syneval-disc:
	python src/main.py syneval \
	    --dynet-autobatch=1 \
	    --dynet-mem=3000 \
	    --model-type=disc-rnng \
	    --checkpoint=${DISC_PATH} \
	    --indir=data/syneval/data/converted \
	    --num-samples=50


# clear temporary models
.PHONY : clean
clean :
	-rm -r models/temp/*
