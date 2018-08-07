import numpy as np
import torch
import torchtext
from torchtext.vocab import FastText
#
# def unk_init(x):
#     return 0.01 * torch.normal(x)
#
# embedding = FastText(unk_init=unk_init)
#
# print(embedding['the'])
# print(embedding['there'])
# print(embedding['UNK'])
# print(embedding['asdbja'])
