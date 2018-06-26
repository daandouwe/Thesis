"""This approach will never work: some list of sequences are longer than others"""

import numpy as np

from get_configs import ences

def grab_actions(sents):
    return [d['actions'] for d in sents]

def unpack(lists):
    return [i for l in lists for i in l]

def compare(pred, gold):
    pred = unpack(grab_actions(pred))
    gold = unpack(grab_actions(gold))
    pred = np.array(pred)
    gold = np.array(gold)
    return np.mean(np.array(pred) == np.array(gold))

data_path = '../tmp/ptb.oracle'
pred_path = 'out/train.predict.txt'

gold = ences(data_path)
pred = get_sentences(pred_path)

n = len(pred)
gold = gold[:n]

print(compare(pred, gold))
