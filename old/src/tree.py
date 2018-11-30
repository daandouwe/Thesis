from datatypes import Item


class Node:
    pass


class InternalNode(Node):
    def __init__(self, item):
        assert isinstance(item, Item), f'invalid item {item:!r}'
        self.item = item
        self.children = list()
        self.is_open_nt = True

    def add_child(self, child):
        assert isinstance(child, Node), f'invalid child node {child:!r}'
        self.children.append(child)

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    def __str__(self):
        return self.linearize()

    def __repr__(self):
        children = [child.item.token for child in self.children]
        return 'InternalNode(head={}, children=({}))'.format(self.item, ', '.join(children))

    def linearize(self, with_tag=True):
        return "({} {})".format(
            self.item, " ".join(child.linearize(with_tag) for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def close(self):
        self.is_open_nt = False


class LeafNode(Node):
    def __init__(self, item, tag='*'):
        assert isinstance(item, Item), f'invalid item {item:!r}'
        assert isinstance(tag, str), f'invalid tag {tag:!r}'
        self.item = item
        self.tag = tag
        self.is_open_nt = False

    def __str__(self):
        return self.linearize()

    def __repr__(self):
        return 'LeafNode({!r}, {:})'.format(self.item, self.tag)

    def linearize(self, with_tag=True):
        if with_tag:
            return "({} {})".format(self.tag, self.item)
        else:
            return "{}".format(self.item)

    def leaves(self):
        yield str(self.item)
