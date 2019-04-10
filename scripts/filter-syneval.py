"""
Filter out all sentences with UNK tokens from the syneval dataset
"""
import os
import json


SYN_DIR = os.path.join('data', 'syneval')
IN_DIR = os.path.join(SYN_DIR, 'data', 'converted')
OUT_DIR = os.path.join(SYN_DIR, 'data', 'filtered')

# supervised vocabulary used by all generative models
VOCAB_PATH = os.path.join('data', 'vocab', 'vocab.json')

# the list of files as produced by convert-syneval.py
FILES = [
    'simple_agrmt',
    'sent_comp',
    'vp_coord',
    'long_vp_coord',
    'prep_anim',
    'prep_inanim',
    'subj_rel',
    'obj_rel_across_anim',
    'obj_rel_across_inanim',
    'obj_rel_no_comp_across_anim',
    'obj_rel_no_comp_across_inanim',
    'obj_rel_no_comp_within_anim',
    'obj_rel_no_comp_within_inanim',
    'obj_rel_within_anim',
    'obj_rel_within_inanim',

    'simple_reflexives',
    'reflexive_sent_comp',
    'reflexives_across',

    'simple_npi_anim',
    'simple_npi_anim_the',
    'simple_npi_inanim',
    'simple_npi_inanim_the',
    'npi_across_anim',
    'npi_across_anim_the',
    'npi_across_inanim',
    'npi_across_inanim_the',
]


def main():

    assert os.path.exists(IN_DIR), 'first convert syneval with `scripts/convert-syneval.py`'
    assert os.path.exists(VOCAB_PATH), 'first build vocabulary with `src/build.py`'

    os.makedirs(OUT_DIR, exist_ok=True)

    files = dict()
    for fname in FILES:

        print(f'Converting `{fname}`...')

        with open(os.path.join(IN_DIR, fname + '.pos')) as f:
            pos_lines = [line.strip() for line in f.readlines()]

        with open(os.path.join(IN_DIR, fname + '.neg')) as f:
            neg_lines = [line.strip() for line in f.readlines()]

        with open(VOCAB_PATH) as f:
            vocab = json.load(f)

        def has_no_unks(line):
            return all(word in vocab for word in line.split())

        pos_lines_filtered, neg_lines_filtered = [], []
        for pos_line, neg_line in zip(pos_lines, neg_lines):
            if has_no_unks(pos_line) and has_no_unks(neg_line):
                pos_lines_filtered.append(pos_line)
                neg_lines_filtered.append(neg_line)

        with open(os.path.join(OUT_DIR, fname + '.pos'), 'w') as f:
            print('\n'.join(pos_lines_filtered), file=f)

        with open(os.path.join(OUT_DIR, fname + '.neg'), 'w') as f:
            print('\n'.join(neg_lines_filtered), file=f)


    print(f'Saved output to `{OUT_DIR}`.')


if __name__ == '__main__':
    main()
