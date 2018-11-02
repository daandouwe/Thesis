import os
import json
from tqdm import tqdm

from actions import SHIFT, REDUCE, NT, GEN, is_nt, is_gen, get_nt, get_word


def get_oracle_dict(oracle):
    oracle_dict = {
        'tree'     : oracle[0],
        'tags'     : oracle[1],
        'original' : oracle[2],
        'lower'    : oracle[3],
        'unked'    : oracle[4],
        'actions'  : oracle[5:]
    }
    return oracle_dict


def get_oracles(path):
    """Chunks the oracle file into sentences."""
    oracles = []
    with open(path) as f:
        oracle = []
        for line in f:
            if line == '\n':
                oracles.append(oracle)
                oracle = []
            else:
                oracle.append(line.rstrip())
        return [get_oracle_dict(oracle) for oracle in oracles if oracle]


def make_generetive(oracles, text):
    for oracle in oracles:
        words = iter(oracle[text].split())
        actions = []
        for action in oracle['actions']:
            if action == 'SHIFT':
                actions.append(GEN(next(words)))
            else:
                actions.append(action)
        oracle['actions'] = actions
    return oracles


class Dictionary:

    def __init__(self, path, text='unked', model='disc'):
        assert text in ('original', 'lower', 'unked'), text
        assert model in ('disc', 'gen'), model

        self.text = text
        self.model = model
        words, actions, nonterminals = self.get_vocab(path)
        self.i2w = words
        self.i2a = actions
        self.i2n = nonterminals
        self.w2i = dict(zip(words, range(len(words))))
        self.a2i = dict(zip(actions, range(len(actions))))
        self.n2i = dict(zip(nonterminals, range(len(nonterminals))))

    def get_vocab(self, path):
        if self.model == 'disc':
            return self.get_disc_vocab(path)
        elif self.model == 'gen':
            return self.get_gen_vocab(path)

    def get_disc_vocab(self, path):
        sentences = get_oracles(path)
        words = list(sorted(set(
            [word for sentence in sentences for word in sentence[self.text].split()])))
        nonterminals = list(sorted(set(
            [get_nt(action) for sentence in sentences for action in sentence['actions'] if is_nt(action)])))
        actions = [SHIFT, REDUCE] + [NT(nt) for nt in nonterminals]
        return words, actions, nonterminals

    def get_gen_vocab(self, path):
        sentences = get_oracles(path)
        words = list(sorted(set(
            [word for sentence in sentences for word in sentence[self.text].split()])))
        nonterminals = list(sorted(set(
            [get_nt(action) for sentence in sentences for action in sentence['actions'] if is_nt(action)])))
        actions = [REDUCE] + [NT(nt) for nt in nonterminals] + [GEN(word) for word in words]
        return words, actions, nonterminals

    def save(self, path):
        path = path + '.json' if not path.endswith('.json') else path
        state = dict(n2i=self.n2i, a2i=self.a2i, w2i=self.w2i)
        with open(path, 'w') as f:
            json.dump(state, f, indent=4)


class Data:

    def __init__(self, path, dictionary, text='unked', model='model'):
        self.text = text
        self.model = model
        self.words = []
        self.actions = []
        self.read(path, dictionary)

    def read(self, path, dictionary):
        oracles = get_oracles(path)
        oracles = make_generetive(oracles, self.text) if self.model == 'gen' else oracles
        for oracle in tqdm(oracles):
            self.words.append(
                [dictionary.w2i[word] for word in oracle[self.text].split()])
            self.actions.append(
                [dictionary.a2i[action] for action in oracle['actions']])

    @property
    def data(self):
        return list(zip(self.words, self.actions))


class Corpus:

    def __init__(self, train_path, dev_path, test_path, text='unked', model='disc'):
        self.dictionary = Dictionary(train_path, text=text, model=model)
        self.train = Data(train_path, self.dictionary, text=text, model=model)
        self.dev = Data(dev_path, self.dictionary, text=text, model=model)
        self.test = Data(test_path, self.dictionary, text=text, model=model)
