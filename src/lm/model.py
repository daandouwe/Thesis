import dynet as dy
import numpy as np

from components.feedforward import Affine, Feedforward


START = '<START>'
STOP = '<STOP>'


class LanguageModel(object):
    def __init__(
            self,
            model,
            word_vocab,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
            dropout,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("LanguageModel")
        self.word_vocab = word_vocab
        self.lstm_dim = lstm_dim

        self.embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))

        self.rnn_builder = dy.VanillaLSTMBuilder(
            lstm_layers,
            word_embedding_dim,
            lstm_dim,
            self.model)

        self.out = Affine(
            self.model, lstm_dim, word_vocab.size)

        self.dropout = dropout
        self.training = True

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    @property
    def num_params(self):
        return sum(np.prod(p.shape()) for p in self.model.parameters_list())

    def train(self):
        self.rnn_builder.set_dropouts(self.dropout, self.dropout)
        self.training = True

    def eval(self):
        self.rnn_builder.disable_dropout()
        self.training = True

    def forward(self, words):

        if self.training:
            words = self.word_vocab.unkify(words)

        rnn = self.rnn_builder.initial_state()

        word_ids = [self.word_vocab.index_or_unk(word) for word in [START] + words]

        prev_embeddings = [self.embeddings[word_id] for word_id in word_ids[:-1]]
        lstm_outputs = rnn.transduce(prev_embeddings)
        logits = self.out(dy.concatenate_to_batch(lstm_outputs))
        nlls = dy.pickneglogsoftmax_batch(logits, word_ids[1:])

        return dy.sum_batches(nlls)


class MultitaskLanguageModel(object):
    def __init__(
            self,
            model,
            word_vocab,
            label_vocab,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
            label_hidden_dim,
            dropout,
            predict_all_spans=False
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("MultitaskLanguageModel")
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim
        self.predict_all_spans = predict_all_spans

        self.embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))

        self.rnn_builder = dy.VanillaLSTMBuilder(
            lstm_layers,
            word_embedding_dim,
            lstm_dim,
            self.model)

        self.out = Affine(
            self.model, lstm_dim, word_vocab.size)

        num_labels = label_vocab.size + 1 if predict_all_spans else label_vocab.size
        self.f_label = Feedforward(
            self.model, lstm_dim, [label_hidden_dim], num_labels)

        self.dropout = dropout
        self.training = True

        self.correct = 0
        self.predicted = 0

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    @property
    def num_params(self):
        return sum(np.prod(p.shape()) for p in self.model.parameters_list())

    def train(self):
        self.rnn_builder.set_dropouts(self.dropout, self.dropout)
        self.training = True

    def eval(self):
        self.rnn_builder.disable_dropout()
        self.training = False

    def forward(self, words, spans=None):
        multitask = spans is not None

        if self.training:
            words = self.word_vocab.unkify(words)

        rnn = self.rnn_builder.initial_state()

        if multitask:
            # need stop token to make span encodings possible (see below)
            word_ids = [self.word_vocab.index_or_unk(word)
                for word in [START] + words + [STOP]]

            prev_embeddings = [self.embeddings[word_id] for word_id in word_ids[:-1]]
            lstm_outputs = rnn.transduce(prev_embeddings)
            logits = self.out(dy.concatenate_to_batch(lstm_outputs))
            nlls = dy.pickneglogsoftmax_batch(logits, word_ids[1:])
            word_nll = dy.sum_batches(nlls)

            # predict label for each possible span (null for nonexistent spans)
            if self.predict_all_spans:
                gold_spans = {(left, right): self.label_vocab.index(label)
                    for left, right, label in spans}

                all_spans = [(left, left + length)
                    for length in range(1, len(words) + 1)
                    for left in range(0, len(words) + 1 - length)]

                label_ids = [gold_spans.get((left, right), self.label_vocab.size)  # last index is for null label
                    for left, right in all_spans]

                # 'lstm minus' features, same as those of the crf parser
                span_encodings = [lstm_outputs[right] - lstm_outputs[left]
                    for left, right in all_spans]

            # only predict labels for existing spans
            else:
                label_ids = [self.label_vocab.index(label) for _, _, label in spans]

                # 'lstm minus' features, same as those of the crf parser
                span_encodings = [lstm_outputs[right] - lstm_outputs[left]
                    for left, right, label in spans]

            logits = self.f_label(dy.concatenate_to_batch(span_encodings))
            nlls = dy.pickneglogsoftmax_batch(logits, label_ids)
            label_nll = dy.sum_batches(nlls)

            # easy proxy to track progress on this task
            self.correct += np.sum(np.argmax(logits.npvalue(), axis=0) == label_ids)
            self.predicted += len(label_ids)

            nll = word_nll + label_nll

        else:
            word_ids = [self.word_vocab.index_or_unk(word)
                for word in [START] + words]

            prev_embeddings = [self.embeddings[word_id] for word_id in word_ids[:-1]]
            lstm_outputs = rnn.transduce(prev_embeddings)
            logits = self.out(dy.concatenate_to_batch(lstm_outputs))
            nlls = dy.pickneglogsoftmax_batch(logits, word_ids[1:])
            nll = dy.sum_batches(nlls)

        return nll
