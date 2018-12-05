def topsort(n):
    """All nodes in a complete forest over n words in topoligical order."""
    for length in range(1, n + 1):
        for left in range(0, n + 1 - length):
            right = left + length
            for label in self.label_vocab.values:
                yield left, right, label

def viterbi(n, tag='*', semiring=ViterbiSemiring):
    chart = {}
    for node in topsort(n):
        left, right, label = node
        label_index = self.label_vocab.index(label)

        label_score = get_label_scores(left, right)[label_index]
        span_score = get_span_score(left, right)
        edge_weight = semiring.product(label_score, span_score)

        if right == left + 1:
            score = edge_weight
            children = [trees.LeafParseNode(left, tag, words[left])]
            subtree = trees.InternalParseNode(label, children)
            chart[node] = score, subtree
        else:
            subtrees = []
            for split in range(left+1, right):

                left_score, left_subtree = max(
                    [chart[left, split, lab] for lab in self.label_vocab.values],
                    key=lambda t: t[0].value())

                right_score, right_subtree = max(
                    [chart[split, right, lab] for lab in self.label_vocab.values],
                    key=lambda t: t[0].value())

                score = semiring.products([
                    edge_weight,
                    left_score,
                    right_score
                ])

                subtrees.append(
                    (score, left_subtree, right_subtree))

            best_score, best_left_subtree, best_right_subtree = max(subtrees, key=lambda t: t[0].value())
            children = [best_left_subtree, best_right_subtree]
            subtree = trees.InternalParseNode(label, children)
            chart[node] = best_score, subtree

    return chart
