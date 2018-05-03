import logging
import argparse

from parser.train import train
from parser.predict import predict

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M')
logger = logging.getLogger(__name__)


def main():

  ap = argparse.ArgumentParser(description="a simple graph-based parser")
  ap.add_argument('--mode', choices=['train', 'predict'], default='train')

  ap.add_argument('--train_path', type=str,
                  default='data/wsj-conllx-3_5_0/train.stanford.conll')
  ap.add_argument('--dev_path', type=str,
                  default='data/wsj-conllx-3_5_0/dev.stanford.conll')
  ap.add_argument('--test_path', type=str,
                  default='data/wsj-conllx-3_5_0/test.stanford.conll')

  ap.add_argument('--n_iters', type=int, default=50000)

  ap.add_argument('--dim', type=int, default=400)
  ap.add_argument('--emb_dim', type=int, default=100)
  ap.add_argument('--pos_emb_dim', type=int, default=100)
  ap.add_argument('--batch_size', type=int, default=5000)

  ap.add_argument('--num_layers', type=int, default=3)
  ap.add_argument('--print_every', type=int, default=10)
  ap.add_argument('--eval_every', type=int, default=100)

  ap.add_argument('--arc_mlp_dim', type=int, default=500)
  ap.add_argument('--lab_mlp_dim', type=int, default=100)

  ap.add_argument('--dropout_p', type=float, default=0.33)
  ap.add_argument('--clip', type=float, default=5.)
  ap.add_argument('--word_dropout_p', type=float, default=0.33)
  ap.add_argument('--pos_dropout_p', type=float, default=0.33)

  ap.add_argument('--optimizer', type=str, default='adam')
  ap.add_argument('--lr', type=float, default=0.002)
  ap.add_argument('--lr_decay', type=float, default=0.75)
  ap.add_argument('--lr_decay_steps', type=float, default=5000)
  ap.add_argument('--adam_beta1', type=float, default=0.9)
  ap.add_argument('--adam_beta2', type=float, default=0.9)
  ap.add_argument('--weight_decay', type=float, default=1e-8)

  ap.add_argument('--seed', type=int, default=42)

  ap.add_argument('--glove', dest='glove', default=True, action='store_true')
  ap.add_argument('--no_glove', dest='glove', default=False, action='store_false')

  ap.add_argument('--uni', dest='bi', default=True, action='store_false',
                  help='use unidirectional encoder')

  cfg = vars(ap.parse_args())
  print("Config:")
  for k, v in cfg.items():
    print("  %12s : %s" % (k, v))

  if cfg['mode'] == 'train':
    train(**cfg)
  elif cfg['mode'] == 'predict':
    predict(**cfg)


if __name__ == '__main__':
  main()
