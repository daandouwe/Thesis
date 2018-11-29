import string

import dynet as dy

from feedforward import Affine
from glove import load_glove


class Embedding:
    """Trainable word embeddings."""
    def __init__(self, model, size, embedding_dim, init='glorot'):

        self.model = model.add_subcollection('Embedding')

        self.size = size
        self.embedding_dim = embedding_dim
        self.embedding = self.model.add_lookup_parameters(
            (size, embedding_dim), init=init)

    def __getitem__(self, index):
        return self(index)

    def __call__(self, index):
        return self.embedding[index]

    @property
    def shape(self):
        return self.embedding.shape


class PretrainedEmbedding:
    """Pretrained word embeddings with optional freezing."""
    def __init__(self, model, size, embedding_dim, vec_dir, ordered_words, type='glove', freeze=True):
        assert type in ('glove',), 'only GloVe supported'

        self.model = model.add_subcollection('PretrainedEmbedding')

        self.size = size
        self.embedding_dim = embedding_dim
        self.freeze = freeze
        self.embedding = self.model.lookup_parameters_from_numpy(
            load_glove(ordered_words, self.embedding_dim, vec_dir))

    def __getitem__(self, index):
        return self(index)

    def __call__(self, index):
        if self.freeze:
            return dy.lookup(self.embedding, index, update=False)
        else:
            return self.embedding[index]

    @property
    def shape(self):
        return self.embedding.shape


class FineTuneEmbedding:
    """Fine-tunes a pretrained Embedding layer.

    Follows the method described in Yoav Goldberg's `Neural
    Network Methods for Natural LanguageProcessing` (p. 117).
    In particular we let the fine-tuned embeddings W' be
    computed as

        W' = W + Δ

    where W are the pre-trained embeddings and Δ the fine-tuning
    weights. W is fixed, and the weights in Δ are initialized
    to zero and trained with an L2 penalty.

    Also: see this https://twitter.com/adveisner/status/896428540538834944
    and this https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting.

    Note: the average norm of the pretrained GloVe embeddings is ~0.03,
    and if, say, we'd like our fine-tuning to be at most a tenth of that,
    e.g. of the order 0.001, then 100 * 0.001 = 0.1 is the additional
    loss we incur. This seems sensible to me. Experiments seem to agree.
    """
    def __init__(self, model, size, embedding_dim, vec_dir, ordered_words, weight_decay=100):

        self.model = model.add_subcollection('FineTuneEmbedding')

        self.size = size
        self.embedding_dim = embedding_dim
        self.weight_decay = weight_decay
        self.embedding = PretrainedEmbedding(
            self.model, size, embedding_dim, vec_dir, ordered_words, freeze=True)
        self.delta = Embedding(self.model, size, embedding_dim, init=0)

    def __getitem__(self, index):
        return self(index)

    def __call__(self, index):
        """Return W'[index] where W' = W + Δ."""
        return self.embedding(index) + self.delta(index)

    def delta_norm(self):
        """Return the (average) L2 norm of Δ."""
        # We average over vocabulary, otherwise we are
        # not consistent accross varying vocab-sizes.
        return dy.sum_elems(dy.squared_norm(self.delta.embedding)) / self.size

    def delta_penalty(self):
        """Return the (average) L2 norm of Δ scaled by the weight-decay term."""
        return self.weight_decay * self.delta_norm()

    @property
    def shape(self):
        return self.embedding.shape


class CharEmbedding:
    """Character LSTM embeddings."""
    VOCAB = list(string.printable)
    UNK = '▁'  # U+2581
    START = '>>'
    STOP = '<<'

    def __init__(self, model, char_embedding_dim, char_lstm_dim, char_lstm_layers, word_embedding_dim):

        self.model = model.add_subcollection('CharEmbedding')

        self.embedding_dim = word_embedding_dim
        self.char_vocab = Vocabulary.fromlist(
            self.VOCAB + [self.UNK, self.START, self.STOP])

        self.char_embeddings = self.model.add_lookup_parameters(
            (char_vocab.size, char_embedding_dim))
        self.char_lstm = dy.BiRNNBuilder(
            char_lstm_layers,
            char_embedding_dim,
            2 * char_lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)
        self.output = Affine(model, 2 * char_lstm_dim, word_embedding_dim)

    def __getitem__(self, word):
        return self(word)

    def __call__(self, word):
        assert isinstance(word, str)
        chars = [char_vocab.index_or_unk(char, self.UNK)
            for char in [self.START] + list(word) + [self.STOP]]
        char_lstm_outputs = self.char_lstm.transduce([
            self.char_embeddings[index] for index in chars])
        char_encoding = self.output(dy.concatenate([
            char_lstm_outputs[-1][:self.char_lstm_dim],
            char_lstm_outputs[0][self.char_lstm_dim:]]))
        return char_encoding

    @property
    def shape(self):
        return (1, self.embedding_dim)
