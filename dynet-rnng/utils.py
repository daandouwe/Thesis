import os
import time
from datetime import datetime

from nltk import Tree


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
        self.time0 = time.time()
        self.previous = time.time()

    def elapsed(self):
        return time.time() - self.time0

    def elapsed_since_previous(self):
        new = time.time()
        elapsed = new - self.previous
        self.previous = new
        return elapsed

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


# TODO: move these functions to a more sensible place.

def unkify(tokens, words_dict):
    final = []
    for token in tokens:
        # only process the train singletons and unknown words
        if len(token.rstrip()) == 0:
            final.append('UNK')
        elif not(token.rstrip() in words_dict):
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
            final.append(result)
        else:
            final.append(token.rstrip())
    return final


def get_actions_no_tags(line):
    """Get actions for a tree without tags.

    Author: Daan van Stigt
    """
    output_actions = []
    line_strip = line.rstrip()
    i = 0
    max_idx = (len(line_strip) - 1)
    while i <= max_idx:
        if line_strip[i] == '(':
            NT = ''
            i += 1
            while line_strip[i] != ' ':
                NT += line_strip[i]
                i += 1
            output_actions.append('NT(' + NT + ')')
        elif line_strip[i] == ')':
             output_actions.append('REDUCE')
             if i == max_idx:
                 break
             i += 1
        else: # it's a terminal symbol
            output_actions.append('SHIFT')
            while line_strip[i] not in (' ', ')'):
                i += 1
        while line_strip[i] == ' ':
            if i == max_idx:
                break
            i += 1
    assert i == max_idx
    return output_actions


def add_dummy_tags(tree, tag='*'):
    """Turns (NP The tagless tree) into (NP (* The) (* tagless) (* tree))."""
    assert isinstance(tree, str), tree
    i = 0
    max_idx = (len(tree) - 1)
    new_tree = ''
    while i <= max_idx:
        if tree[i] == '(':
            new_tree += tree[i]
            i += 1
            while tree[i] != ' ':
                new_tree += tree[i]
                i += 1
        elif tree[i] == ')':
            new_tree += tree[i]
            if i == max_idx:
                break
            i += 1
        else: # it's a terminal symbol
            new_tree += f'({tag} '
            while tree[i] not in (' ', ')'):
                new_tree += tree[i]
                i += 1
            new_tree += ')'
        while tree[i] == ' ':
            if i == max_idx:
                break
            new_tree += tree[i]
            i += 1
    assert i == max_idx, i
    return new_tree


def substitute_leaves(tree, new_leaves):
    assert isinstance(tree, str), tree
    assert all(isinstance(leaf, str) for leaf in new_leaves), new_leaves
    old_leaves = Tree.fromstring(tree).leaves()
    message = f'inconsistent lengths:\nOld: {list(old_leaves)}\nNew: {list(new_leaves)}'
    assert len(old_leaves) == len(list(new_leaves)), message
    new_leaves = iter(new_leaves)
    i = 0
    max_idx = (len(tree) - 1)
    new_tree = ''
    while i <= max_idx:
        assert tree[i] != ' '
        if tree[i] == '(':
            new_tree += tree[i]
            i += 1
            while tree[i] != ' ':
                new_tree += tree[i]
                i += 1
        elif tree[i] == ')':
            new_tree += tree[i]
            if i == max_idx:
                break
            i += 1
        else: # it's a terminal symbol
            while tree[i] not in (' ', ')'):
                i += 1
            new_tree += next(new_leaves)
        # Skip whitespace.
        while tree[i] == ' ':
            if i == max_idx:
                break
            new_tree += tree[i]
            i += 1
    assert i == max_idx, i
    return new_tree
