import numpy as np

from get_configs import get_sentences

data_path = '../tmp/ptb.oracle'
pred_path = 'out/train.predict.txt'

gold = get_sentences(data_path)
pred = get_sentences(pred_path)

n = len(pred)

gold = gold[:n]
pred['actions'] == gold[]
acc = np.mean(pred['actions'] == gold['actions'])
print(acc)
