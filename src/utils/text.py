
BRACKETS = [
    'LRB', 'RRB',
    'LSB', 'RSB',
    'RCB', 'RCB'
]


def replace_quotes(words):
    """Replace quotes following PTB convention"""
    assert isinstance(words, list), words
    assert all(isinstance(word, str) for word in words), words

    replaced = []
    found_left_double, found_left_single = False, False
    for word in words:
        if word == '"':
            if found_left_double:
                found_left_double = False
                replaced.append("''")
            else:
                found_left_double = True
                replaced.append("``")
        elif word == "'":
            if found_left_double:
                found_left_double = False
                replaced.append("'")
            else:
                found_left_double = True
                replaced.append("`")
        else:
            replaced.append(word)
    return replaced


def replace_brackets(words):
    """Replace brackets following PTB convention"""
    assert isinstance(words, list), words
    assert all(isinstance(word, str) for word in words), words

    replaced = []
    for word in words:
        if word == '(':
            replaced.append('LRB')
        elif word == '{':
            replaced.append('LCB')
        elif word == '[':
            replaced.append('LSB')
        elif word == ')':
            replaced.append('RRB')
        elif word == '}':
            replaced.append('RCB')
        elif word == ']':
            replaced.append('RSB')
        else:
            replaced.append(word)
    return replaced


def unkify(token, words_dict):
    """Elaborate UNKing following parsing tradition."""
    if len(token.rstrip()) == 0:
        final = 'UNK'
    else:
        numCaps = 0
        hasDigit = False
        hasDash = False
        hasLower = False
        for char in token.rstrip():
            if char.isdigit():
                hasDigit = True
            elif char == '-':
                hasDash = True
            elif char.isalpha():
                if char.islower():
                    hasLower = True
                elif char.isupper():
                    numCaps += 1
        result = 'UNK'
        lower = token.rstrip().lower()
        ch0 = token.rstrip()[0]
        if ch0.isupper():
            if numCaps == 1:
                result = result + '-INITC'
                if lower in words_dict:
                    result = result + '-KNOWNLC'
            else:
                result = result + '-CAPS'
        elif not(ch0.isalpha()) and numCaps > 0:
            result = result + '-CAPS'
        elif hasLower:
            result = result + '-LC'
        if hasDigit:
            result = result + '-NUM'
        if hasDash:
            result = result + '-DASH'
        if lower[-1] == 's' and len(lower) >= 3:
            ch2 = lower[-2]
            if not(ch2 == 's') and not(ch2 == 'i') and not(ch2 == 'u'):
                result = result + '-s'
        elif len(lower) >= 5 and not(hasDash) and not(hasDigit and numCaps > 0):
            if lower[-2:] == 'ed':
                result = result + '-ed'
            elif lower[-3:] == 'ing':
                result = result + '-ing'
            elif lower[-3:] == 'ion':
                result = result + '-ion'
            elif lower[-2:] == 'er':
                result = result + '-er'
            elif lower[-3:] == 'est':
                result = result + '-est'
            elif lower[-2:] == 'ly':
                result = result + '-ly'
            elif lower[-3:] == 'ity':
                result = result + '-ity'
            elif lower[-1] == 'y':
                result = result + '-y'
            elif lower[-2:] == 'al':
                result = result + '-al'
        final = result
    return final
