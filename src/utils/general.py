import os
import time
from datetime import datetime


def ceil_div(a, b):
    return ((a - 1) // b) + 1


def get_folders(args):
    """Create paths for logging and checkpoints."""
    if args.disable_subdir:
        subdir, logdir, checkdir, outdir = (
            None, args.logdir, args.checkdir, args.outdir)
    else:
        subdir = get_subdir_string(args, with_params=False)  # Too many parameters for folder.
        logdir = os.path.join(args.logdir, subdir)
        checkdir = os.path.join(args.checkdir, subdir)
        outdir = os.path.join(args.outdir, subdir)
    return subdir, logdir, checkdir, outdir


def get_subdir_string(args, with_params=True):
    """Returns a concatenation of a date and timestamp."""
    date = time.strftime('%Y%m%d')
    timestamp = time.strftime('%H%M%S')
    return f'{date}_{timestamp}'


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
    if isinstance(expression, dy.Expression):
        return expression.value()
    else:  # already detached
        return expression


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


class Config(object):
    """Class that loads hyperparameters from json file into attributes"""

    def __init__(self, source):
        """
        Args:
            source: path to json file or dict
        """
        self.source = source

        if type(source) is dict:
            self.__dict__.update(source)
        elif type(source) is list:
            for s in source:
                self.load_json(s)
        else:
            self.load_json(source)


    def load_json(self, source):
        with open(source) as f:
            data = json.load(f)
            self.__dict__.update(data)


    def save(self, dir_name):
        init_dir(dir_name)
        if type(self.source) is list:
            for s in self.source:
                c = Config(s)
                c.save(dir_name)
        elif type(self.source) is dict:
            json.dumps(self.source, indent=4)
        else:
            copyfile(self.source, dir_name + self.export_name)
