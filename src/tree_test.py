from datatypes import Item


class Node:
    pass


class InternalNode(Node):
    def __init__(self, item):
        assert isinstance(item, Item)
        self.item = item
        self.children = list()
        self.is_open_nt = True

    def add_child(self, child):
        assert isinstance(child, Node)
        self.children.append(child)

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    def __str__(self):
        children = [child.item.token for child in self.children]
        return 'InternalNode({}, ({}))'.format(self.item, ', '.join(children))

    def linearize(self):
        return "({} {})".format(
            self.item, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def close(self):
        self.is_open_nt = False


class LeafNode(Node):
    def __init__(self, item, tag='*'):
        assert isinstance(item, Item)
        self.item = item
        self.tag = tag
        self.is_open_nt = False

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


if __name__ == '__main__':
    pass
