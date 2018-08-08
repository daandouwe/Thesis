from collections.abc import Sequence

import torch

from data import Item, Action

class Node:
    pass

class InternalNode(Node):
    def __init__(self, item, head):
        # assert isinstance(item, Action)
        self.item = item
        self.head = head
        self.children = tuple()

    def add_child(self, child):
        assert isinstance(child, Node)
        assert child
        self.children = (*self.children, child) # new extended tuple

    def __str__(self):
        children = [child.item for child in self.children]
        return 'InternalNode({}, ({}))'.format(self.item, ', '.join(children))

    def linearize(self):
        return "({} {})".format(
            self.item, " ".join(child.linearize() for child in self.children))
            # self.item.symbol, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    @property
    def label(self):
        # return self.item.symbol
        return self.item

class LeafNode(Node):
    def __init__(self, item, head, tag='*'):
        # assert isinstance(item, Item)
        self.item = item
        self.head = head
        self.tag = tag

    def __str__(self):
        return 'LeafNode({!r}, {:})'.format(self.item, self.tag)

    def linearize(self):
        return "({} {})".format(self.tag, self.item)

    def leaves(self):
        yield self

class Tree:
    """A tree that is constructed top-down."""
    def __init__(self):
        self.root = None # The root node
        self.current_node = None # Keep track of the current node
        self.num_open_nonterminals = 0

    def make_root(self, item):
        print(f'making root `{item}`')
        self.root = self.current_node = InternalNode(item, None)

    def open_nonterminal(self, item):
        print(f'opening nonterminal `{item}`')
        self.num_open_nonterminals += 1
        # If current node is a nonterminal.
        head = self.get_current_head()
        node = InternalNode(item, head)
        head.add_child(node)
        self.current_node = node

    def make_leaf(self, item):
        print(f'making leaf `{item}`')
        head = self.get_current_head()
        node = LeafNode(item, head)
        head.add_child(node)
        self.current_node = node

    def close_nonterminal(self):
        print(f'reducing under {self.get_current_head()}')
        self.num_open_nonterminals -= 1
        head = self.get_current_head()
        children = head.children
        self.current_node = self.get_current_head().head # Move two nodes up!
        return head, children

    def linearize(self):
        return self.root.children[0].linearize()

    def get_current_head(self):
        if isinstance(self.current_node, InternalNode):
            return self.current_node
        else:
            return self.current_node.head


if __name__ == '__main__':
    pass
