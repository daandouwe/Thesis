import os

from PYEVALB.scorer import Scorer
from get_vocab import get_sentences

def oracle2tree(sent):
    """Returns a linearize tree from a list of actions in an oracle file.

    Args:
        sent: a dictionary as returned by get_sent_dict in get_configs.py
    """
    actions = sent['actions']
    words = sent['upper'].split()
    tags = sent['tags'].split()
    gold_tree = sent['tree'][2:] # remove the hash
    tree = ''
    # reverse words:
    words = words[::-1]
    tags = tags[::-1]
    for i, a in enumerate(actions):
        if a == 'SHIFT':
            w = words.pop()
            t = tags.pop()
            tree += '({} {}) '.format(t, w)
        elif a == 'REDUCE':
            tree += ') '
        else:
            nt = a[3:-1] # a is NT(X), and we select only X
            tree += '({} '.format(nt)
    return tree, gold_tree

if __name__ == '__main__':
    outdir = 'out'

    dev_path = os.path.join(outdir, 'dev')
    dev_pred = get_sentences(dev_path + '.pred.oracle')
    pred_path = dev_path + '.pred.trees'
    gold_path = dev_path + '.gold.trees'
    result_path = dev_path + '.result'
    with open(dev_path + '.pred.trees', 'w') as f:
        with open(dev_path + '.gold.trees', 'w') as g:
            for sent in dev_pred:
                pred_tree, gold_tree = oracle2tree(sent)
                print(pred_tree, file=f)
                print(gold_tree, file=g)
    scorer = Scorer()
    scorer.evalb(gold_path, pred_path, result_path)

    test_path = os.path.join(outdir, 'test')
    test_pred = get_sentences(test_path + '.pred.oracle')
    pred_path = test_path + '.pred.trees'
    gold_path = test_path + '.gold.trees'
    result_path = test_path + '.result'
    with open(test_path + '.pred.trees', 'w') as f:
        with open(test_path + '.gold.trees', 'w') as g:
            for sent in test_pred:
                pred_tree, gold_tree = oracle2tree(sent)
                print(pred_tree, file=f)
                print(gold_tree, file=g)
    scorer = Scorer()
    scorer.evalb(gold_path, pred_path, result_path)
