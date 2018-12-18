"""
Evaluate tagging accuracy on the sentiment treebank.
"""

import sys

from tree import fromstring


def main(gold, pred):

    def read(path):
        with open(path) as f:
            trees = [fromstring(line.strip()) for line in f.readlines() if line.strip()]
        return trees

    gold_trees = read(gold)
    pred_trees = read(pred)
    assert len(gold_trees) == len(pred_trees)

    root_acc = sum(int(g.label) == int(p.label) for g, p in zip(gold_trees, pred_trees)) \
        / len(gold_trees)
    all_acc = sum(g_label == p_label
                  for gold_tree, pred_tree in zip(gold_trees, pred_trees)
                  for g_label, p_label in zip(gold_tree.labels(), pred_tree.labels())) \
        / sum(map(len, (tree.labels() for tree in gold_trees)))

    print(f'Root accuracy {root_acc:.2%}')
    print(f'All accuracy {all_acc:.2%}')


if __name__ == '__main__':
    if not len(sys.argv) == 3:
        exit('usage: eval.py <gold> <pred>')
    else:
        main(sys.argv[1], sys.argv[2])
