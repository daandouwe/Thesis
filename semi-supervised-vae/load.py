import torch

from mnist import MNISTCached

data_loaders = setup_data_loaders(MNISTCached, args.use_cuda, args.batch_size, sup_num=args.sup_num)
