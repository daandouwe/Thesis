import os
import sys
from glob import glob


def filter_folders(folders, name, include=[], exclude=[]):
    filtered = []
    for folder in folders:
        folder = folder.split('_')
        if folder[0] == name:
            if all(name in folder for name in include) and all(not name in folder for name in exclude):
                filtered.append(folder)
    if not filtered:
        filtered.append([''])

    return filtered

def get_max_dev(folders):
    assert all(isinstance(folder, list) for folder in folders)
    return '_'.join(max(folders, key=lambda l: l[-1]))

def get_min_dev(folders):
    assert all(isinstance(folder, list) for folder in folders)
    return '_'.join(max(folders, key=lambda l: l[-1]))

def main():
    model_dir = 'models'
    temp_dir = 'temp'

    folders = [d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
        and d != temp_dir]

    lm_dir = get_min_dev(
        filter_folders(folders, 'lm', exclude=['vocab']))

    lm_vocab_dir = get_min_dev(
        filter_folders(folders, 'lm', include=['vocab']))

    os.environ['LM_PATH'] = os.path.join(model_dir, lm_dir)
    os.environ['LM_VOCAB_PATH'] = os.path.join(model_dir, lm_vocab_dir)

    print(os.environ)

if __name__ == '__main__':
    main()
