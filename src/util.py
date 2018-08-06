import os
import time

class Timer:
    """A simple timer to use during training."""
    def __init__(self):
        self.time0 = time.time()

    def elapsed(self):
        time1 = time.time()
        elapsed = time1 - self.time0
        self.time0 = time1
        return elapsed

def clock_time(s):
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return int(h), int(m), int(s)

def get_parameter_string(args):
    """Returns an identification string based on arguments in args.

    Example:
        `batch_size_25_lr_1e-4`

    Note:
        Some of the values in args are paths, e.g. `--data ../tmp/ptb`.
        These contain `/` and cannot be used in the directory string.
        We filter these out.
    """
    keys = []
    args = vars(args)
    for i, key in enumerate(args):
        val = args[key]
        if isinstance(val, str):
            if not '/' in val: # filter out paths
                keys.append(key)
        else:
            keys.append(key)
    params = [f'{key}_{args[key]}' for key in sorted(keys)]
    return '_'.join(params)

def get_subdir_string(args, with_params=False):
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

def write_args(args, positional=('mode',)):
    """Writes args to a file to be later used as input in the command line.

    Only works for arguments with double dash, e.g. --verbose, and
    positional arguments are not printed.

    Arguments:
        positional (tuple): the positional arguments in args that are skipped

    TODO: There should be a more robust way to do this!
    """
    with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
        for k, v in vars(args).items():
            if k not in positional: # skip positional arguments
                print(f'--{k}={v}', file=f)

def write_losses(args, losses):
    path = os.path.join(args.logdir, 'loss.csv')
    with open(path, 'w') as f:
        print('loss', file=f)
        for loss in losses:
            print(loss, file=f)

def make_folders(args):
    # Create folders for logging and checkpoints
    subdir = get_subdir_string(args)
    logdir   = os.path.join(args.root, 'log', subdir)
    checkdir = os.path.join(args.root, 'checkpoints', subdir)
    outdir   = os.path.join(args.root, 'out', subdir)

    os.mkdir(logdir)
    os.mkdir(checkdir)
    os.mkdir(outdir)

    logfile   = os.path.join(logdir, 'train.log')
    checkfile = os.path.join(checkdir, 'model.pt')
    outfile   = os.path.join(outdir, 'train.predict.txt')

    # Add folders and dirs to args
    args.outdir     = outdir
    args.outfile    = outfile
    args.logdir     = logdir
    args.logfile    = logfile
    args.checkdir   = checkdir
    args.checkfile  = checkfile
    # Save the arguments.
    write_args(args)
