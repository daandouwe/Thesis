evalb:
	get-evalb.sh

data:
	get-ptb.sh && get-unlabeled.sh

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
	python build.py \
    @configs/vocab/supervised.txt

sup-lowercase-vocab:
	python build.py \
    @configs/vocab/supervised.txt \
		--lowercase

semisup-vocab:
	python build.py \
    @configs/vocab/semisupervised.txt

semisup-lowercase-vocab:
	python build.py \
    @configs/vocab/semisupervised.txt \
		--lowercase

# Train a supervised model
train-disc:
	python main.py train \
	    --model-path-base=models/disc-rnng \
	    @configs/data/supervised.txt \
	    @configs/model/disc-rnng.txt \
	    @configs/training/sgd.txt

train-gen:
	python main.py train \
	    --model-path-base=models/gen-rnng \
	    @configs/data/supervised.txt \
	    @configs/model/gen-rnng.txt \
	    @configs/training/sgd.txt \
	    @configs/proposal/disc-rnng.txt

train-crf:
	python main.py train \
	    --model-path-base=models/crf \
	    @configs/data/supervised.txt \
	    @configs/model/crf.txt \
	    @configs/training/adam.txt \

train-disc-vocab: semisup-vocab
	python main.py train \
	    --model-path-base=models/disc-rnng_vocab=semisup \
	    @configs/vocab/semisupervised.txt \
	    @configs/data/supervised.txt \
	    @configs/model/disc-rnng.txt \
	    @configs/training/sgd.txt

train-gen-vocab: semisup-vocab
	python main.py train \
	    --model-path-base=models/gen-rnng_vocab=semisup \
	    @configs/vocab/semisupervised.txt \
	    @configs/data/supervised.txt \
	    @configs/model/gen-rnng.txt \
	    @configs/training/sgd.txt \
	    @configs/proposal/disc-rnng.txt

train-crf-vocab: semisup-vocab
	python main.py train \
	    --model-path-base=models/crf_vocab=semisup \
	    @configs/vocab/semisupervised.txt \
	    @configs/data/supervised.txt \
	    @configs/model/crf.txt \
	    @configs/training/adam.txt

# Finetune models semisupervised
semisup-rnng:
	python main.py semisup \
	    --model-path-base=models/semisup_post=disc \
	    @configs/data/semisupervised.txt \
	    @configs/model/semisup-rnng \
	    @configs/training/adam.txt \
	    @configs/baseline/argmax.txt

semisup-crf:
	python main.py semisup \
	    --model-path-base=models/semisup_post=crf \
	    @configs/data/semisupervised.txt \
	    @configs/model/semisup-crf \
	    @configs/training/adam.txt \
	    @configs/baseline/argmax.txt

semisup-rnng-vocab:
	python main.py semisup \
	    --model-path-base=models/semisup_post=disc_vocab=semisup \
	    @configs/vocab/semisupervised.txt \
	    @configs/data/semisupervised.txt \
	    @configs/model/semisup-rnng \
	    @configs/training/adam.txt \
	    @configs/baseline/argmax.txt

semisup-crf-vocab:
	python main.py semisup \
	    --model-path-base=models/semisup_post=crf_vocab=semisup \
	    @configs/vocab/semisupervised.txt \
	    @configs/data/semisupervised.txt \
	    @configs/model/semisup-crf \
	    @configs/training/adam.txt \
	    @configs/baseline/argmax.txt

# Eval
syneval-crf:
	python main.py syneval \
    @configs/model/semisup-rnng.txt

syneval-crf:
	python main.py syneval \
    @configs/model/semisup-crf.txt
