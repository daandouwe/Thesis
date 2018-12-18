disc: train-disc
gen: train-gen
crf: train-crf

disc-semisup-vocab: build-semisup train-disc-semisup-vocab
gen-semisup-vocab: build-semisup train-gen-semisup-vocab
crf-semisup-vocab: build-semisup train-crf-semisup-vocab

disc-semisup-lowercase-vocab: build-semisup-lowercase train-disc-semisup-vocab
gen-semisup-lowercase-vocab: build-semisup-lowercase train-gen-semisup-vocab
crf-semisup-lowercase-vocab: build-semisup-lowercase train-crf-semisup-vocab

# Build a vocabulary
build-sup:
	python build.py \
		@configs/vocab/supervised.txt

build-sup-lowercase:
	python build.py \
		@configs/vocab/supervised.txt \
		--lowercase

build-semisup:
	python build.py \
		@configs/vocab/semisupervised.txt

build-semisup-lowercase:
	python build.py \
		@configs/vocab/semisupervised.txt \
		--lowercase

# Train a supervised model
train-disc:
	python main.py train \
		@configs/data/supervised.txt \
		@configs/model/disc-rnng.txt \
		@configs/training/sgd.txt

train-gen:
	python main.py train \
		@configs/data/supervised.txt \
		@configs/model/gen-rnng.txt \
		@configs/training/sgd.txt \
		@configs/proposals/disc-rnng.txt

train-crf:
	python main.py train \
		@configs/data/supervised.txt \
		@configs/model/crf.txt \
		@configs/training/adam.txt

train-disc-semisup-vocab:
	python main.py train \
		@configs/vocab/semisupervised.txt \
		@configs/data/supervised.txt \
		@configs/model/disc-rnng.txt \
		@configs/training/sgd.txt

train-gen-semisup-vocab:
	python main.py train \
		@configs/vocab/supervised.txt \
		@configs/data/supervised.txt \
		@configs/model/gen-rnng.txt \
		@configs/training/sgd.txt \
		@configs/proposals/disc-rnng.txt

train-crf-semisup-vocab:
	python main.py train \
		@configs/vocab/supervised.txt \
		@configs/data/supervised.txt \
		@configs/model/crf.txt \
		@configs/training/adam.txt


# Finetune models semisupervised
semisup-rnng:
	python main.py semisup \
		@configs/data/semisupervised.txt \
		@configs/models/semisup-rnng \
		@configs/training/adam.txt \
		--use-argmax-baseline

semisup-crf:
	python main.py semisup \
		@configs/data/semisupervised.txt \
		@configs/models/semisup-crf \
		@configs/training/adam.txt \
		--use-argmax-baseline

semisup-rnng-vocab:
	python main.py semisup \
		@configs/vocab/semisupervised.txt \
		@configs/data/semisupervised.txt \
		@configs/models/semisup-rnng \
		@configs/training/adam.txt \
		--use-argmax-baseline

semisup-crf-vocab:
	python main.py semisup \
		@configs/vocab/semisupervised.txt \
		@configs/data/semisupervised.txt \
		@configs/models/semisup-crf \
		@configs/training/adam.txt \
		--use-argmax-baseline

# Eval
syneval-crf-proposal:
	python main.py predict \
		--syneval \
		@configs/models/semisup-crf
