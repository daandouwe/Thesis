from actions import SHIFT, NT, GEN, REDUCE


class Node:
    pass


class InternalNode(Node):
    def __init__(self, label):
        assert isinstance(label, str), label

        self.label = label
        self.children = []

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

    def tags(self):
        for child in self.children:
            yield from child.tags()

    def labels(self):
        return [self.label] + [label for child in self.children for label in child.labels()]

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


class LeafNode(Node):
    def __init__(self, word, label='*'):
        assert isinstance(word, str), word
        assert isinstance(label, str), label

        self.word = word
        self.label = label

    def __str__(self):
        return self.linearize()

    def __repr__(self):
        return 'LeafNode({!r}, {:})'.format(self.word, self.label)

    def linearize(self, with_tag=True):
        if with_tag:
            return "({} {})".format(self.label, self.word)
        else:
            return "{}".format(self.word)

    def leaves(self):
        yield self.word

    def tags(self):
        yield self.label

    def labels(self):
        return []

    def gen_oracle(self):
        return [GEN(self.word)]

    def disc_oracle(self):
        return [SHIFT]

    def substitute_leaves(self, words):
        self.word = next(words)


def fromstring(tree):
    """Return a tree from a string."""
    assert isinstance(tree, str), tree
    assert len(tree) > 0

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
                node = InternalNode(label)
                node.add_children(children)
                trees.append(node)
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

    return trees.pop()


def add_dummy_tags(tree, tag='*'):
    """Turns '(NP The tagless tree)' into '(NP (* The) (* tagless) (* tree))'."""
    assert isinstance(tree, str), tree
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
