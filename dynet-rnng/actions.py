
SHIFT = 'SHIFT'

REDUCE = 'REDUCE'

def NT(nt):
    return f'NT({nt})'

def GEN(word):
    return f'GEN({word})'

def is_nt(action):
    return action.startswith('NT(') and action.endswith(')')

def is_gen(action):
    return action.startswith('GEN(') and action.endswith(')')

def get_nt(action):
    assert is_nt(action)
    return action[3:-1]

def get_word(action):
    assert is_gen(action)
    return action[4:-1]
