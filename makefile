evalb:
	scripts/get-evalb.sh 

data:
	mkdir data && scripts/get-ptb.sh && scripts/get-unlabeled.sh

disc: train-disc
gen: train-gen
crf: train-crf

disc-vocab: semisup-vocab train-disc-vocab
gen-vocab: semisup-vocab train-gen-vocab
crf-vocab: semisup-vocab train-crf-vocab

disc-lowercase-vocab: semisup-lowercase-vocab train-disc-vocab
gen-lowercase-vocab: semisup-lowercase-vocab train-gen-vocab
crf-lowercase-vocab: semisup-lowercase-vocab train-crf-vocab

# Build a vocabulary
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
	    @src/configs/proposal/disc-rnng.txt

train-crf:
	python src/main.py train \
	    --model-path-base=models/crf \
	    @src/configs/data/supervised.txt \
	    @src/configs/model/crf.txt \
	    @src/configs/training/adam.txt \

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

# Finetune models semisupervised
semisup-rnng:
	python src/main.py semisup \
	    --model-path-base=models/semisup_post=disc \
	    @src/configs/data/semisupervised.txt \
	    @src/configs/model/semisup-rnng \
	    @src/configs/training/adam.txt \
	    @src/configs/baseline/argmax.txt

semisup-crf:
	python src/main.py semisup \
	    --model-path-base=models/semisup_post=crf \
	    @src/configs/data/semisupervised.txt \
	    @src/configs/model/semisup-crf \
	    @src/configs/training/adam.txt \
	    @src/configs/baseline/argmax.txt

semisup-rnng-vocab:
	python src/main.py semisup \
	    --model-path-base=models/semisup_post=disc_vocab=semisup \
	    @src/configs/vocab/semisupervised.txt \
	    @src/configs/data/semisupervised.txt \
	    @src/configs/model/semisup-rnng \
	    @src/configs/training/adam.txt \
	    @src/configs/baseline/argmax.txt

semisup-crf-vocab:
	python src/main.py semisup \
	    --model-path-base=models/semisup_post=crf_vocab=semisup \
	    @src/configs/vocab/semisupervised.txt \
	    @src/configs/data/semisupervised.txt \
	    @src/configs/model/semisup-crf \
	    @src/configs/training/adam.txt \
	    @src/configs/baseline/argmax.txt

# Eval
syneval-crf:
	python src/main.py syneval \
	    @src/configs/model/semisup-rnng.txt

syneval-rnng:
	python src/main.py syneval \
	    @src/configs/model/semisup-rnng.txt
