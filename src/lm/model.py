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

        # TODO: use the embedding classes
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
        else:
            words = self.word_vocab.process(words)

        rnn = self.rnn_builder.initial_state()

        nll = dy.zeros(1)
        for prev, word in zip([START] + words[:-1], words):
            input = self.embeddings[self.word_vocab.index_or_unk(prev)]
            rnn = rnn.add_input(input)
            logits = self.out(rnn.output())
            nll += dy.pickneglogsoftmax(
                logits, self.word_vocab.index_or_unk(word))
        return nll


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
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("MultitaskLanguageModel")
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim

        # TODO: use the embedding classes
        self.embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))

        self.rnn_builder = dy.VanillaLSTMBuilder(
            lstm_layers,
            word_embedding_dim,
            lstm_dim,
            self.model)

        self.out = Affine(
            self.model, lstm_dim, word_vocab.size)
        self.f_label = Feedforward(
            # self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size)
            self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size + 1)  # when predicting all spans

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
        else:
            words = self.word_vocab.process(words)

        rnn = self.rnn_builder.initial_state()

        if multitask:
            lstm_outputs = []
            word_nll = dy.zeros(1)
            for prev, word in zip([START] + words, words + [STOP]):  # need stop to make span prediction work
                input = self.embeddings[self.word_vocab.index_or_unk(prev)]
                rnn = rnn.add_input(input)
                hidden = rnn.output()
                logits = self.out(hidden)
                word_nll += dy.pickneglogsoftmax(
                    logits, self.word_vocab.index_or_unk(word))
                lstm_outputs.append(hidden)

            # # predict labeled spans as 'scaffold' task
            # label_nll = dy.zeros(1)
            # for left, right, label in spans:
            #     hidden = dy.concatenate(
            #         [lstm_outputs[left], lstm_outputs[right]])
            #     logits = self.f_label(hidden)
            #     label_id = self.label_vocab.index(label)
            #     label_nll += dy.pickneglogsoftmax(
            #         logits, label_id)
            #     # easy track progress on this task
            #     self.correct += np.argmax(logits.value()) == label_id
            #     self.predicted += 1

            # Predict tag for each span (null for nonexistent spans)
            # Runs at about 2/3 of the speed on the above
            spans = {(left, right): self.label_vocab.index(label)
                for left, right, label in spans}

            label_nll = dy.zeros(1)
            for length in range(1, len(words) + 1):
                for left in range(0, len(words) + 1 - length):
                    right = left + length
                    label_id = spans.get((left, right), self.label_vocab.size)  # last index is for null label
                    hidden = dy.concatenate(
                        [lstm_outputs[left], lstm_outputs[right]])
                    logits = self.f_label(hidden)
                    label_nll += dy.pickneglogsoftmax(
                        logits, label_id)
                    self.correct += np.argmax(logits.value()) == label_id
                    self.predicted += 1

            nll = word_nll + label_nll
        else:
            nll = dy.zeros(1)
            for prev, word in zip([START] + words[:-1], words):
                input = self.embeddings[self.word_vocab.index_or_unk(prev)]
                rnn = rnn.add_input(input)
                logits = self.out(rnn.output())
                nll += dy.pickneglogsoftmax(
                    logits, self.word_vocab.index_or_unk(word))
        return nll
