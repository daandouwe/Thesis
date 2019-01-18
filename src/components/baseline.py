import dynet as dy

from .feedforward import Feedforward, Affine
from utils.general import blockgrad


class FeedforwardBaseline:

    def __init__(self, model, model_type, lstm_dim, hidden_dim=128):
        assert model_type in ('disc', 'crf'), model_type

        self.model = model.add_subcollection('FeedforwardBaseline')

        self.model_type = model_type
        self.lstm_dim = lstm_dim
        self.hidden_dim = hidden_dim

        self.gating = Feedforward(
            self.model, lstm_dim, [hidden_dim], 1)
        self.feedforward = Feedforward(
            self.model, lstm_dim, [hidden_dim], 1)

    def forward(self, words, parser):

        word_indices = [parser.word_vocab.index_or_unk(word) for word in words]
        embeddings = [parser.word_embedding[word_id] for word_id in word_indices]

        # use the rnn from the posterior model to create contextual embeddings
        if self.model_type == 'crf':
            lstm_outputs = parser.lstm.transduce(embeddings)

        elif self.model_type == 'disc':
            lstm = parser.buffer_encoder.rnn_builder.initial_state()
            lstm_outputs = lstm.transduce(reversed(embeddings))  # in reverse! see class Buffer for why

        # detach the embeddings (gradients undesired for this part of the baseline)
        lstm_outputs = [dy.inputTensor(blockgrad(encoding)) for encoding in lstm_outputs]

        gates = [dy.logistic(self.gating(output)) for output in lstm_outputs]
        encodings = [dy.cmult(gate, output) for gate, output in zip(gates, lstm_outputs)]

        return self.feedforward(dy.esum(encodings))
