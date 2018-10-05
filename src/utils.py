import os
import time

from nltk import Tree


def get_folders(args):
    # Create folders for logging and checkpoints
    subdir = get_subdir_string(args, with_params=False)  # Too many parameters for folder.
    logdir = os.path.join(args.root, args.logdir, subdir)
    checkdir = os.path.join(args.root, args.checkdir, subdir)
    outdir = os.path.join(args.root, args.outdir, subdir)
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


# TODO: move these to a sensible place.

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
    message = f'inconsistent lengths:\nOld: {list(old_leaves)}\nNew {list(new_leaves)}'
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
