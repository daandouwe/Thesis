import collections.abc

DUMMY = '@'

class TreebankNode(object):
    pass

class InternalTreebankNode(TreebankNode):
    def __init__(self, label, children):
        assert isinstance(label, str)
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, TreebankNode) for child in children)
        assert children
        self.children = tuple(children)

    def linearize(self):
        return "({} {})".format(
            self.label, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self, index=0):
        tree = self
        sublabels = [self.label]

        # collapse unary chains
        while len(tree.children) == 1 and isinstance(
                tree.children[0], InternalTreebankNode):
            tree = tree.children[0]
            sublabels.append(tree.label)

        children = []
        for child in tree.children:
            children.append(child.convert(index=index))
            index = children[-1].right

        return InternalParseNode(tuple(sublabels), children)

class LeafTreebankNode(TreebankNode):
    def __init__(self, tag, word):
        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def linearize(self):
        # return "({} {})".format(self.tag, self.word)
        return self.word

    def leaves(self):
        yield self

    def convert(self, index=0):
        return LeafParseNode(index, self.tag, self.word)

class ParseNode(object):
    pass

class InternalParseNode(ParseNode):
    def __init__(self, label, children):
        assert isinstance(label, tuple)
        assert all(isinstance(sublabel, str) for sublabel in label)
        assert label
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, ParseNode) for child in children)
        assert children
        assert len(children) > 1 or isinstance(children[0], LeafParseNode)
        assert all(
            left.right == right.left
            for left, right in zip(children, children[1:]))
        self.children = tuple(children)

        self.left = children[0].left
        self.right = children[-1].right

    def spans(self):
        return [(self.left, self.right, self.label)] + \
            [span for child in self.children for span in child.spans()]

    def labels(self):
        return [self.label] + \
            [label for child in self.children for label in child.labels()]

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self):
        children = [child.convert() for child in self.children]
        tree = InternalTreebankNode(self.label[-1], children)
        for sublabel in reversed(self.label[:-1]):
            tree = InternalTreebankNode(sublabel, [tree])
        return tree

    def linearize(self):
        span = str(self.left) + '-' + str(self.right)
        label = '+'.join(self.label)
        return "({} {})".format(
            label + ':' + span, " ".join(child.linearize() for child in self.children))

    def binarize(self):
        if len(self.children) == 1:
            assert isinstance(self.children[0], LeafParseNode)
            return InternalParseNode(self.label, self.children)
        if len(self.children) == 2:
            return InternalParseNode(
                self.label, [child.binarize() for child in self.children])
        else:
            left = self.children[0]
            right = InternalParseNode((DUMMY,), self.children[1:])
            return InternalParseNode(
                self.label, [left.binarize(), right.binarize()])

    def unbinarize(self):
        # absorb empty children until none are empty
        children = self.children
        while not all(not child.is_dummy for child in children):
            new = ()
            for child in children:
                if child.is_dummy:
                    for grandchild in child.children:
                        new += (grandchild,)
                else:
                    new += (child,)
            children = new
        # recursively unbinarize the children
        children = tuple((child.unbinarize() for child in children))
        return InternalParseNode(self.label, children)

    @property
    def is_dummy(self):
        return self.label == (DUMMY,)


class LeafParseNode(ParseNode):
    def __init__(self, index, tag, word):
        assert isinstance(index, int)
        assert index >= 0
        self.left = index
        self.right = index + 1

        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def spans(self):
        return []

    def labels(self):
        return []

    def leaves(self):
        yield self

    def convert(self):
        return LeafTreebankNode(self.tag, self.word)

    def binarize(self):
        return self

    def linearize(self):
        return self.word

    def unbinarize(self):
        return self

    @property
    def is_dummy(self):
        return False


def load_trees(path, strip_top=True):
    with open(path) as infile:
        tokens = infile.read().replace("(", " ( ").replace(")", " ) ").split()

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
                trees.append(InternalTreebankNode(label, children))
            else:
                word = tokens[index]
                index += 1
                trees.append(LeafTreebankNode(label, word))

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1

        return trees, index

    trees, index = helper(0)
    assert index == len(tokens)

    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label == "TOP":
                assert len(tree.children) == 1
                trees[i] = tree.children[0]

    return trees


if __name__ == '__main__':
    # testing
    from nltk import Tree
    from nltk.draw.tree import TreeView

    treebank = load_trees(
        '/Users/daan/data/ptb-benepar/22.auto.clean', strip_top=True)

    tree = treebank[0]

    tree = tree.convert()
    binary = tree.binarize()
    unbinary = tree.binarize().unbinarize()
    assert unbinary.linearize() == tree.linearize()

    tree = Tree.fromstring(tree.linearize())
    binary = Tree.fromstring(binary.linearize())
    unbinary = Tree.fromstring(unbinary.linearize())

    TreeView(tree)._cframe.print_to_file('tree.ps')
    TreeView(binary)._cframe.print_to_file('binary.ps')
    TreeView(unbinary)._cframe.print_to_file('unbinary.ps')
