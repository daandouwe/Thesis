from datatypes import Action

SHIFT = Action('SHIFT', Action.SHIFT_INDEX)

REDUCE = Action('REDUCE', Action.REDUCE_INDEX)

def GEN(word):
    token = f'GEN({word.token})'
    return Action(token, word.index, word.embedding, word.encoding)

def NT(nt):
    token = f'NT({nt.token})'
    return Action(token, nt.index, nt.embedding, nt.encoding)
