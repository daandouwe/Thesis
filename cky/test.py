def incoming(left, right):
    """Return all incoming nodes to a node that spans left to right."""
    for split in range(left+1, right):
        for label in range(3):
            yield (left, split, label)
            yield (split, right, label)
