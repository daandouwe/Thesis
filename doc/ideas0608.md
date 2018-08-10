# Todo list

## Datasets
- [ ] There are alternatives to the PTB: see [this stackoverflow](https://stackoverflow.com/questions/8949517/is-there-any-treebank-for-free).
  * The [TED treebank](https://ahcweb01.naist.jp/resource/tedtreebank/) is by Graham Neubig!
  * The [Stanford sentiment treebank](https://nlp.stanford.edu/sentiment/).

## Data
- [x] Use an Item class two wrap indices and tokens together:
    * `Item(word, index)`, then `word = item.word` and `index = item.index`.
    * Use a tokenizer to process input from terminal?
- [ ] `Item` object is nice! `Action` object is pretty ugly though... Fix this!
- [x] Implement a `Tree` class to store the created trees during parsing.
    * This way we can store the represenations of the StackLSTM, which is useful for inspection.
    * Then we can pass a `Tree` object from `model.parse` directly.

## Embeddings
- [x] Ask Wilker if he knows a nice default character embedding method (e.g. convolutions). Answer: The character convolutions I already have implemented.
- [x] Glove embeddings.
- [x] FastText.
- [ ] Elmo embeddings: figure out what the fuss is about, and incorporate them (only discriminative!).
    * Compute embeddings once: save to file, then load like it were glove. But then sentence_id -> token_id -> vector.
    * Will make everything moooooore slow.

## Model
- [x] Make prediction 2-step: first predict from `(open_nonterminal, shift, reduce)`, then if `open_nonterminal`, predict `NT(X)`.
    * This makes generative model easier: predict from `(open_nonterminal, generate, reduce)` then if `generate` predict word.

## Prediction / Eval
- [x] Put some effort into cleaning this up.
    * Prediction should produce as final output a file with trees! Not an oracle...
    * Eval should *only* use EVALB to compute F1 score, nothing else!
    * Use the original evalb?

## Decoding
- [ ] Add beam-search to prediction: see if this improves f1.
- [ ] Ancestral sampler decoder.

## Training
- [x] Use TensorboadX to keep track of training.
- [x] Training from: https://github.com/nikitakit/self-attentive-parser/blob/master/src/main.py.
- [ ] Look into trainer class from https://github.com/allenai/allennlp/blob/master/allennlp/training/trainer.py.


## Visualization
- [ ] A live broswer demo to inspect the trees loaded from textfile.
  * http://christos-c.com/treeviewer/
- [ ] A live demo that predicts tree for input and visualizes it in browser.
- [ ] A demo that shows how the stack embedding changes per update step.
