import json

data_path  = '../data/ptb/dep/json/dev-stanford-raw.json'

data = json.load(open(data_path))

print(len(data))

print
