import os

import dynet as dy

from model import DiscRNNG
from data import Corpus


def build_corpus(args):
    corpus = Corpus(
        train_path=os.path.join(args.data, 'train/ptb.train.oracle'),
        dev_path=os.path.join(args.data, 'dev/ptb.dev.oracle'),
        test_path=os.path.join(args.data, 'test/ptb.test.oracle'),
        text_type=args.text_type,
        rnng_type=args.rnng_type
    )
    return corpus.dictionary

def build_model(args, model, dictionary):
    return DiscRNNG(
        model=model,
        dictionary=dictionary,
        num_words=dictionary.num_words,
        num_nt=dictionary.num_nt,
        word_emb_dim=args.word_emb_dim,
        nt_emb_dim=args.nt_emb_dim,
        action_emb_dim=args.action_emb_dim,
        stack_lstm_dim=args.stack_lstm_dim,
        buffer_lstm_dim=args.buffer_lstm_dim,
        history_lstm_dim=args.history_lstm_dim,
        stack_lstm_layers=args.lstm_layers,
        buffer_lstm_layers=args.lstm_layers,
        history_lstm_layers=args.lstm_layers,
        composition=args.composition,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        use_glove=args.use_glove,
        glove_dir=args.glove_dir,
        fine_tune_embeddings=args.fine_tune_embeddings,
        freeze_embeddings=args.freeze_embeddings,
    )

def main(args):

    model = dy.ParameterCollection()
    dictionary = build_corpus(args)
    rnng = build_model(args, model, dictionary)
    model.populate(args.checkpoint)

    dy.save(self.model_checkpoint_path, [rnng])
