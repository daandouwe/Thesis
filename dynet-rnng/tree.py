from actions import SHIFT, NT, GEN, REDUCE


class Node:
    pass


class InternalNode(Node):
    def __init__(self, label):
        assert isinstance(label, str), label

        self.label = label
        self.children = []
        self.is_open_nt = True

    def add_child(self, child):
        assert isinstance(child, Node), child
        self.children.append(child)

    def add_children(self, children):
        assert isinstance(children, list), children
        for child in children:
            self.add_child(child)

    def __str__(self):
        return self.linearize()

    def __repr__(self):
        children = [child.label for child in self.children]
        return 'InternalNode(head={}, children=({}))'.format(self.label, ', '.join(children))

    def linearize(self, with_tag=True):
        return "({} {})".format(
            self.label, " ".join(child.linearize(with_tag) for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def close(self):
        self.is_open_nt = False

    def gen_oracle(self):
        """Top-down generative oracle."""
        return [NT(self.label)] + \
               [action for child in self.children for action in child.gen_oracle()] + \
               [REDUCE]

    def disc_oracle(self):
        """Top-down discriminative oracle."""
        return [NT(self.label)] + \
               [action for child in self.children for action in child.disc_oracle()] + \
               [REDUCE]

    @staticmethod
    def fromstring(tree):
        return fromstring(tree)


class LeafNode(Node):
    def __init__(self, label, tag='*'):
        assert isinstance(label, str), label
        assert isinstance(tag, str), tag

        self.label = label
        self.tag = tag
        self.is_open_nt = False

    def __str__(self):
        return self.linearize()

    def __repr__(self):
        return 'LeafNode({!r}, {:})'.format(self.label, self.tag)

    def linearize(self, with_tag=True):
        if with_tag:
            return "({} {})".format(self.tag, self.label)
        else:
            return "{}".format(self.label)

    def leaves(self):
        yield str(self.label)

    def gen_oracle(self):
        return [GEN(self.label)]

    def disc_oracle(self):
        return [SHIFT]


def fromstring(tree):
    """Return an InternalNode tree from a string tree."""
    assert isinstance(tree, str), tree

    tokens = tree.replace("(", " ( ").replace(")", " ) ").split()

    def helper(index):
        trees = []

        while index < len(tokens) and tokens[index] == "(":
            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]
            index += 1

            if tokens[index] == "(":
                children, index = helper(index)
                # FIXME: I want to change to __init__(self, label, children=[])
                # but this change gives a bizar error.
                # trees.append(InternalNode(label, children))
                node = InternalNode(label)
                node.add_children(children)
                trees.append(node)
            else:
                word = tokens[index]
                index += 1
                trees.append(LeafNode(label, word))

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1

        return trees, index

    trees, index = helper(0)
    assert index == len(tokens)

    return trees.pop()


if __name__ == '__main__':
    S = InternalNode('S')
    NP = InternalNode('NP')
    VP = InternalNode('VP')
    The = LeafNode('The')
    cat = LeafNode('cat')
    eats = LeafNode('eats')
    period = LeafNode('.')

    NP.add_children([The, cat])
    VP.add_child(eats)
    S.add_children([NP, VP, period])

    tree = S
    print(tree.linearize(with_tag=False))
    actions = tree.gen_oracle()
    print(actions)

    trees = fromstring(tree.linearize(with_tag=True))
    print(trees)
