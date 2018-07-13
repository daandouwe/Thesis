import spacy

from newdata import Corpus

nlp = spacy.load('en')

corpus = Corpus(data_path='../tmp/ptb', textline='lower')

words = corpus.dictionary.i2w[:100]
doc = nlp(' '.join(words))
# for w in words:
    # print(w)
for token in doc:
    # print(token.text, token.lemma_, token.norm, token.shape)
    print(token.text, token.norm_, token.shape_, token.prefix_)
