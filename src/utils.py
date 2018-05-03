import time

class Timer:
    """
    A simple timer to use during training.
    """
    def __init__(self):
        self.time0 = time.time()

    def elapsed(self):
        time1 = time.time()
        elapsed = time1 - self.time0
        self.time0 = time1
        return elapsed

def get_parameter_string(args):
    """
    Returns a string of type `batch_size_25_lr_1e-4` based on arguments in args.
    Note: Some of the values in args are paths, e.g. `--data ../tmp/ptb`.
    These contain `/` and cannot be used in the directory string.
    We filter these out.
    """
    keys = []
    args = vars(args)
    for key in args:
        val = args[key]
        if isinstance(val, str):
            if not '/' in val: # filter out paths
                keys.append(key)
        else:
            keys.append(key)
    params = ['{}_{}'.format(key, args[key]) for key in sorted(keys)]
    return '_'.join(params)

def get_subdir_string(args):
    """
    Returns a string of the convention 20170524_192314_batch_size_25_lr_1e-4/
    """
    date = time.strftime('%Y%m%d')
    timestamp = time.strftime('%H%M%S')
    params = get_parameter_string(args)
    return '{}_{}_{}'.format(date, timestamp, params)
