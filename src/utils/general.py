import os
import shutil
import time
import random
from datetime import datetime

import dynet as dy
from nltk import Tree


def ceil_div(a, b):
    return ((a - 1) // b) + 1


def get_subdir_string():
    """Returns a concatenation of a date and timestamp."""
    date = time.strftime('%Y%m%d')
    timestamp = time.strftime('%H%M%S')
    return f'{date}_{timestamp}'


def get_folders(args, subdir=None):
    """Create paths for logging and checkpoints."""
    if subdir is None:
        # make a new subdirectory
        subdir = os.path.join('models', 'temp', get_subdir_string())
        # avoid overwriting when multiple jobs are submitted simultaneously
        while os.path.exists(subdir):
            time.sleep(random.uniform(1, 10))
            subdir = os.path.join('models', 'temp', get_subdir_string())
    logdir = os.path.join(subdir, 'log')
    outdir = os.path.join(subdir, 'output')
    vocabdir = os.path.join(subdir, 'vocab')
    checkdir = os.path.join(subdir)
    return subdir, logdir, checkdir, outdir, vocabdir


def move_to_final_folder(subdir, model_path_base, dev_fscore):
    """Move `subdir` to `model_path_base_dev=dev_fscore`.

    Example:
            `models/temp/20181220_181640` -> `models/disc-rnng_dev=89.43`
        when
            subdir = `models/temp/20181220_181640`
            model_path_base = `models/disc-rnng`
    """
    final_path = model_path_base + '_dev=' + str(dev_fscore)
    # avoid overwriting existing folders
    while os.path.exists(final_path):
        final_path += '0'  # if needed: dev=87.46 -> dev=87.460 -> dev=87.4600 etc.
    print(f'Moving folder `{subdir}` to `{final_path}`...')
    shutil.move(subdir, final_path)


def write_args(args, logdir, positional=('mode',)):
    """Writes args to a file to be later used as input in the command line."""
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


def blockgrad(expression):
    """Detach a dynet expression from the computation graph"""
    # if isinstance(expression, dy.Expression):
    #     return expression.value()
    # else:  # already detached
    #     return expression
    return expression.value()


def load_model(dir, name='model'):
    path = os.path.join(dir, name)
    model = dy.ParameterCollection()
    [parser] = dy.load(path, model)
    return parser


def load_semisup_models(dir):
    joint_model = load_model(dir, name='joint-model')
    post_model = load_model(dir, name='post-model')
    return joint_model, post_model


def is_tree(line):
    """Simple `oracle` to see if line is a tree."""
    assert isinstance(line, str), line
    try:
        Tree.fromstring(line)
        return True
    except ValueError:
        return False


class Timer:
    """A simple timer to use during training."""
    def __init__(self):
        self.start = time.time()
        self.previous = time.time()

    def elapsed(self):
        return time.time() - self.start

    def elapsed_epoch(self):
        return time.time() - self.previous

    def eta(self, current, total):
        remaining = total - current
        speed = current / self.elapsed_epoch()
        return remaining / speed

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

    def format_elapsed_epoch(self):
        return self.format(self.elapsed_epoch())

    def format_eta(self, current, total):
        return self.format(self.eta(current, total))

    def new_epoch(self):
        self.previous = time.time()
