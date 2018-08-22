import torch
from pympler import muppy, tracker

from datatypes import Word, Nonterminal, Action
from tree_test import LeafNode, InternalNode

def print_memory(message):
    print(message)
    all_objects = muppy.get_objects()
    tensors = muppy.filter(all_objects, Type=torch.Tensor)
    lists = muppy.filter(all_objects, Type=list)
    print(f'objects {len(all_objects):,}')
    print(f'tensors {len(tensors):,}') # Growing at each step
    print(f'not tensors {len(all_objects) - len(tensors):,}')
    print()
