import torch

from data import Item, Action

class Node:
    pass

class InternalNode(Node):
    def __init__(self, item, head):
        assert isinstance(item, Item)
        assert isinstance(head, Node) or isinstance(head, type(None)) # Root node has head None
        self.item = item
        self.head = head
        self.children = tuple()

    def add_child(self, child):
        assert isinstance(child, Node)
        assert child
        self.children = (*self.children, child) # new extended tuple

    def __str__(self):
        children = [child.item.token for child in self.children]
        return 'InternalNode({}, ({}))'.format(self.item, ', '.join(children))

    def linearize(self):
        return "({} {})".format(
            self.item, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

class LeafNode(Node):
    def __init__(self, item, head, tag='*'):
        assert isinstance(item, Item)
        assert isinstance(head, Node)
        self.item = item
        self.head = head
        self.tag = tag

    def __str__(self):
        return 'LeafNode({!r}, {:})'.format(self.item, self.tag)

    def linearize(self):
        if self.tag == '*':
            return "({} {})".format(self.tag, self.item)
            # return "{}".format(self.item)
        else:
            return "({} {})".format(self.tag, self.item)

    def leaves(self):
        yield self

class Tree:
    """A tree that is constructed top-down."""
    def __init__(self):
        self.root = None # The root node
        self.current_node = None # Keep track of the current node
        self.num_open_nonterminals = 0

    def __str__(self):
        return self.linearize()

    def make_root(self, item):
        node = InternalNode(item, None)
        self.root = node
        self.current_node = node

    def make_leaf(self, item):
        head = self.get_current_head()
        node = LeafNode(item, head)
        head.add_child(node)
        self.current_node = node

    def open_nonterminal(self, item):
        self.num_open_nonterminals += 1
        # If current node is a nonterminal.
        head = self.get_current_head()
        node = InternalNode(item, head)
        head.add_child(node)
        self.current_node = node

    def close_nonterminal(self):
        self.num_open_nonterminals -= 1
        head = self.get_current_head()
        children = head.children
        # Move two nodes up from current leaf since we closed the head
        self.current_node = head.head
        return head, children

    def get_current_head(self):
        if isinstance(self.current_node, InternalNode):
            return self.current_node
        else: # current node is a LeafNode
            return self.current_node.head

    def linearize(self):
        assert self.current_node.children, 'no nonterminals opened yet'
        return self.root.children[0].linearize()

    @property
    def last_closed_nonterminal(self):
        assert self.current_node.children, 'no nonterminals opened yet'
        return self.current_node.children[-1]

    @property
    def start(self):
        return not self.root.children

    @property
    def finished(self):
        return self.current_node is self.root and not self.start

if __name__ == '__main__':
    pass
