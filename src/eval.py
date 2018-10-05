#!/usr/bin/env python
import os
import re
import subprocess


def evalb(evalb_dir, pred_path, gold_path, result_path, ignore_error=10000):
    """Use EVALB to score trees."""
    evalb_dir = os.path.expanduser(evalb_dir)
    assert os.path.exists(evalb_dir), f'Do you have EVALB installed at {evalb_dir}?'
    evalb_exec = os.path.join(evalb_dir, "evalb")
    command = '{} {} {} -e {} > {}'.format(
        evalb_exec,
        pred_path,
        gold_path,
        ignore_error,
        result_path
    )
    subprocess.run(command, shell=True)
    # Read result path and get F-sore.
    with open(result_path) as f:
        for line in f:
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore = float(match.group(1))
                return fscore
    return -1
