import os
import psutil

import torch
from pympler import muppy, tracker

from datatypes import Item, Word, Nonterminal, Action
from tree import Node, LeafNode, InternalNode


def print_memory(message):
    print(message)
    all_objects = muppy.get_objects()
    tensors = muppy.filter(all_objects, Type=torch.Tensor)
    lists = muppy.filter(all_objects, Type=list)
    print(f'objects {len(all_objects):,}')
    print(f'tensors {len(tensors):,}')  # Growing at each step
    print(f'not tensors {len(all_objects) - len(tensors):,}')
    print()


def get_added_memory(prev_mem):
    process = psutil.Process(os.getpid())
    cur_mem = process.memory_info().rss / 10**6
    add_mem = cur_mem - prev_mem
    return cur_mem, add_mem


def get_num_objects():
    """
    Not increasing:
        list
        Item
        Node
        InternalNode
        floats (only the loss that we save in a list)
    Increasing:
        tensor (same as Variable)
        int
        str
    """
    all_objects = muppy.get_objects()
    tensors = muppy.filter(all_objects, Type=torch.Tensor)
    strings = muppy.filter(all_objects, Type=str)
    variables = muppy.filter(all_objects, Type=torch.autograd.Variable)
    ints = muppy.filter(all_objects, Type=int)
    return len(all_objects), len(tensors), len(strings), len(ints)


def get_tensors():
    all_objects = muppy.get_objects()
    tensors = muppy.filter(all_objects, Type=torch.Tensor)
    return tensors
