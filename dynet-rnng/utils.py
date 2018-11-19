import os
import time
from datetime import datetime


def ceil_div(a, b):
    return ((a - 1) // b) + 1


def get_folders(args):
    # Create folders for logging and checkpoints
    if args.disable_subdir:
        subdir, logdir, checkdir, outdir = (
            None, args.logdir, args.checkdir, args.outdir)
    else:
        subdir = get_subdir_string(args, with_params=False)  # Too many parameters for folder.
        logdir = os.path.join(args.logdir, subdir)
        checkdir = os.path.join(args.checkdir, subdir)
        outdir = os.path.join(args.outdir, subdir)
    return subdir, logdir, checkdir, outdir


def get_parameter_string(args):
    """Returns an identification string based on arguments in args.

    Example:
        `batch_size_25_lr_1e-4`

    Note:
        Some of the values in args are paths, e.g. `--data ../data/ptb`.
        These contain `/` and cannot be used in the directory string.
        We filter these out.
    """
    keys = []
    args = vars(args)
    for i, key in enumerate(args):
        val = args[key]
        if isinstance(val, str):
            if not '/' in val:  # filter out paths
                keys.append(key)
        else:
            keys.append(key)
    params = [f'{key}_{args[key]}' for key in sorted(keys)]
    return '_'.join(params)


def get_subdir_string(args, with_params=True):
    """Returns a concatenation of a date and timestamp and parameters.

    if with_params:
        20170524_192314_batch_size_25_lr_1e-4/
    else:
        20170524_192314
    """
    date = time.strftime('%Y%m%d')
    timestamp = time.strftime('%H%M%S')
    # now = datetime.now()
    # timestamp = now.strftime('%H%M%S.%f')
    if with_params:
        params = get_parameter_string(args)
        return f'{date}_{timestamp}_{params}'
    else:
        return f'{date}_{timestamp}'


def write_args(args, logdir, positional=('mode',)):
    """Writes args to a file to be later used as input in the command line.

    Only works for arguments with double dash, e.g. --verbose, and
    positional arguments are not printed.

    Arguments:
        positional (tuple): the positional arguments in args that are skipped

    TODO: There should be a more robust way to do this!
    """
    with open(os.path.join(logdir, 'args.txt'), 'w') as f:
        for k, v in vars(args).items():
            if k not in positional: # skip positional arguments
                print(f'--{k}={v}', file=f)


def write_losses(args, losses):
    path = os.path.join(args.logdir, 'loss.csv')
    with open(path, 'w') as f:
        print('loss', file=f)
        for loss in losses:
            print(loss, file=f)


class Timer:
    """A simple timer to use during training."""
    def __init__(self):
        self.start = time.time()
        self.previous = time.time()

    def elapsed(self):
        return time.time() - self.start

    def elapsed_since_previous(self):
        new = time.time()
        elapsed = new - self.previous
        self.previous = new
        return elapsed

    def reset(self):
        self.start = time.time()
        self.previous = time.time()

    def clock_time(self, seconds):
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)
        return int(days), int(hours), int(minutes), int(seconds)

    def format(self, seconds):
        days, hours, minutes, seconds = self.clock_time(seconds)
        elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
        if days > 0:
            elapsed_string = "{}d{}".format(days, elapsed_string)
        return elapsed_string

    def format_elapsed(self):
        return self.format(self.elapsed())


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
