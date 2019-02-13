import collections.abc

from rnng.parser.actions import SHIFT, NT, GEN, REDUCE


TOP = 'TOP'
DUMMY = '@'


class Node(object):
    pass


class InternalNode(Node):
    def __init__(self, label, children):
        assert isinstance(label, str)
        assert isinstance(children, list)

        self.label = label
        self.children = children

    def add_child(self, child):
        assert isinstance(child, Node)
        self.children.append(child)

    def add_children(self, children):
        assert isinstance(children, list)
        for child in children:
            self.add_child(child)

    def __str__(self):
        return self.linearize()

    def __repr__(self):
        children = [child.label for child in self.children]
        return 'InternalNode(head={}, children=({}))'.format(
            self.label, ', '.join(children))

    def linearize(self, with_tag=True):
        return '({} {})'.format(
            self.label, ' '.join(child.linearize(with_tag) for child in self.children))

    def leaves(self):
        return [leaf for child in self.children for leaf in child.leaves()]

    def tags(self):
        return [tag for child in self.children for tag in child.tags()]

    def words(self):
        return [word for child in self.children for word in child.words()]

    def labels(self):
        return [self.label] + \
            [label for child in self.children for label in child.labels()]

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

    def substitute_leaves(self, words):
        for child in self.children:
            child.substitute_leaves(words)

    def convert(self, index=0):
        tree = self
        sublabels = [self.label]

        # collapse unary chains
        while len(tree.children) == 1 and isinstance(
                tree.children[0], InternalNode):
            tree = tree.children[0]
            sublabels.append(tree.label)

        children = []
        for child in tree.children:
            children.append(child.convert(index=index))
            index = children[-1].right

        return InternalSpanNode(tuple(sublabels), children)

    def cnf(self):
        return self.convert().binarize()


class LeafNode(Node):
    def __init__(self, word, label='*'):
        assert isinstance(word, str), word
        assert isinstance(label, str), label

        self.word = word
        self.label = label

    def __str__(self):
        return self.linearize()

    def __repr__(self):
        return 'LeafNode({}, {})'.format(self.word, self.label)

    def linearize(self, with_tag=True):
        if with_tag:
            return '({} {})'.format(self.label, self.word)
        else:
            return '{}'.format(self.word)

    def leaves(self):
        return [self]

    def words(self):
        return [self.word]

    def tags(self):
        return [self.label]

    def labels(self):
        return []

    def gen_oracle(self):
        return [GEN(self.word)]

    def disc_oracle(self):
        return [SHIFT]

    def substitute_leaves(self, words):
        self.word = next(words)

    def convert(self, index=0):
        return LeafSpanNode(index, self.label, self.word)


class SpanNode(object):
    pass


class InternalSpanNode(SpanNode):
    def __init__(self, label, children):
        assert isinstance(label, tuple)
        assert all(isinstance(sublabel, str) for sublabel in label)
        assert label
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, SpanNode) for child in children)
        assert children
        assert len(children) > 1 or isinstance(children[0], LeafSpanNode)
        assert all(
            left.right == right.left
            for left, right in zip(children, children[1:]))
        self.children = tuple(children)

        self.left = children[0].left
        self.right = children[-1].right

    @property
    def is_dummy(self):
        return self.label == (DUMMY,)

    def spans(self):
        return [(self.left, self.right, self.label)] + \
            [span for child in self.children for span in child.spans()]

    def labels(self):
        return [self.label] + \
            [label for child in self.children for label in child.labels()]

    def leaves(self):
        return [leaf for child in self.children for leaf in child.leaves()]

    def tags(self):
        return [tag for child in self.children for tag in child.tags()]

    def words(self):
        return [word for child in self.children for word in child.words()]

    def convert(self):
        children = [child.convert() for child in self.children]
        tree = InternalNode(self.label[-1], children)
        for sublabel in reversed(self.label[:-1]):
            tree = InternalNode(sublabel, [tree])
        return tree

    def linearize(self, with_tag=True):
        span = str(self.left) + '-' + str(self.right)
        label = '+'.join(self.label)
        return '({} {})'.format(
            label + ':' + span, ' '.join(child.linearize(with_tag) for child in self.children))

    def binarize(self):

        def expand(node):
            if isinstance(node, LeafSpanNode):
                return InternalSpanNode((DUMMY,), [node])
            return node

        if len(self.children) == 1:
            assert isinstance(self.children[0], LeafSpanNode)
            return InternalSpanNode(self.label, self.children)
        if len(self.children) == 2:
        	return InternalSpanNode(
                self.label, [expand(child).binarize() for child in self.children])
        else:
            left = expand(self.children[0])
            right = InternalSpanNode((DUMMY,), self.children[1:])
            return InternalSpanNode(
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
        return InternalSpanNode(self.label, children)

    def un_cnf(self):
        return self.unbinarize().convert()


class LeafSpanNode(SpanNode):
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

    def words(self):
        yield self.word

    def convert(self):
        return LeafNode(self.word, self.tag)

    def linearize(self, with_tag=True):
        if with_tag:
            return '({} {})'.format(self.tag, self.word)
        else:
            return '{}'.format(self.word)

    def binarize(self):
        return self

    def unbinarize(self):
        return self

    @property
    def is_dummy(self):
        return False


def fromstring(tree, strip_top=True):
    """Return a tree from a string."""
    assert isinstance(tree, str), tree
    assert len(tree) > 0, tree

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
                trees.append(InternalNode(label, children))
            else:
                word = tokens[index]
                index += 1
                trees.append(LeafNode(word, label))

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1

        return trees, index

    trees, index = helper(0)
    assert index == len(tokens)

    tree = trees.pop()
    if strip_top:
        if tree.label == TOP:
            assert len(tree.children) == 1
            tree = tree.children[0]

    return tree


def add_dummy_tags(tree, tag='*'):
    """Turns '(NP The tagless tree)' into '(NP (* The) (* tagless) (* tree))'."""
    assert isinstance(tree, str), tree
    assert len(tree) > 0, tree

    i = 0
    max_idx = (len(tree) - 1)
    new_tree = ''
    while i <= max_idx:
        if tree[i] == '(':
            new_tree += tree[i]
            i += 1
            while tree[i] != ' ':
                new_tree += tree[i]
                i += 1
        elif tree[i] == ')':
            new_tree += tree[i]
            if i == max_idx:
                break
            i += 1
        else: # it's a terminal symbol
            new_tree += '(' + tag + ' '
            while tree[i] not in (' ', ')'):
                new_tree += tree[i]
                i += 1
            new_tree += ')'
        while tree[i] == ' ':
            if i == max_idx:
                break
            new_tree += tree[i]
            i += 1
    assert i == max_idx, i
    return new_tree


def uncollapse(spans):
    """Uncollapse the unary chains in a list of spans.

    Example:
        turns (1, 4, ('S', 'NP',)) into (1, 4, ('S',)) and (1, 4, ('NP',))
    """
    new_spans = []
    for left, right, labels in spans:
        if len(labels) > 1:
            for label in labels:
                new_spans.append((left, right, (label,)))
        else:
            new_spans.append((left, right, labels))
    return new_spans
