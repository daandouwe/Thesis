from PYEVALB import scorer
from get_configs import get_sentences

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
    pred_oracle_path = 'out/train.predict.txt'
    pred = get_sentences(pred_oracle_path)

    pred_path = 'out/ptb.pred'
    gold_path = 'out/ptb.gold'
    result_path = 'out/ptb.result'
    f = open(pred_path, 'w')
    g = open(gold_path, 'w')
    for sent in pred:
        pred_tree, gold_tree = oracle2tree(sent)
        print(pred_tree, file=f)
        print(gold_tree, file=g)
    f.close()
    g.close()
    scorer = scorer.Scorer()
    scorer.evalb(gold_path, pred_path, result_path)
