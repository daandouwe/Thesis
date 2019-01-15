"""
Convert the syneval dataset from pickled python tuples of the form (grammatical, ungrammatical)
to separate text-files: one grammatical and one ungrammatical, named respectively *.pos and *.neg.

This eases inspection of the dataset, and allows us to more easily produce
syntactic parses of the sentences.

Note: the sentences with negative polarity items (npi) instead come in triples of the form:

    'few shows that the parents love will ever get old'    (grammatical)
    'the shows that few parents love will ever get old'    (ungrammatical)
    'many shows that the parents love will ever get old'   (ungrammatical)

This nuisance is solved by creating four files:
    *.pos / *.neg           ('few'/'many')
    *_the.pos / *_the.neg   ('few'/'the')
"""
import os
import pickle


SYN_DIR = os.path.join('data', 'syneval')
IN_DIR = os.path.join(SYN_DIR, 'data', 'templates')
OUT_DIR = os.path.join(SYN_DIR, 'data', 'converted')

FILES = [
    'long_vp_coord.pickle',
    'obj_rel_across_anim.pickle',
    'obj_rel_across_inanim.pickle',
    'obj_rel_no_comp_across_anim.pickle',
    'obj_rel_no_comp_across_inanim.pickle',
    'obj_rel_no_comp_within_anim.pickle',
    'obj_rel_no_comp_within_inanim.pickle',
    'obj_rel_within_anim.pickle',
    'obj_rel_within_inanim.pickle',
    'prep_anim.pickle',
    'prep_inanim.pickle',
    'reflexive_sent_comp.pickle',
    'reflexives_across.pickle',
    'sent_comp.pickle',
    'simple_agrmt.pickle',
    'simple_reflexives.pickle',
    'subj_rel.pickle',
    'vp_coord.pickle',
]


NPI_FILES = [
    'simple_npi_anim.pickle',
    'simple_npi_inanim.pickle',
    'npi_across_anim.pickle',
    'npi_across_inanim.pickle',
]


def main():

    for fname in FILES:
        with open(os.path.join(IN_DIR, fname), 'rb') as f:
            categories = pickle.load(f)

        pairs = [(pos, neg) for tuples in categories.values() for pos, neg in tuples]
        pos, neg = zip(*pairs)

        base, _ = fname.split('.')
        with open(os.path.join(OUT_DIR, base + '.pos'), 'w') as f:
            print('\n'.join(pos), file=f)
        with open(os.path.join(OUT_DIR, base + '.neg'), 'w') as f:
            print('\n'.join(neg), file=f)


    for fname in NPI_FILES:
        with open(os.path.join(IN_DIR, fname), 'rb') as f:
            categories = pickle.load(f)

        base, _ = fname.split('.')

        # These start with 'the'.
        pairs = [(pos, neg) for tuples in categories.values() for pos, neg, _ in tuples]
        pos, neg = zip(*pairs)
        with open(os.path.join(OUT_DIR, base + '_the' + '.pos'), 'w') as f:
            print('\n'.join(pos), file=f)
        with open(os.path.join(OUT_DIR, base + '_the' + '.neg'), 'w') as f:
            print('\n'.join(neg), file=f)

        # These start with 'most', 'many' etc.
        pairs = [(pos, neg) for tuples in categories.values() for pos, _, neg in tuples]
        pos, neg = zip(*pairs)
        with open(os.path.join(OUT_DIR, base + '.pos'), 'w') as f:
            print('\n'.join(pos), file=f)
        with open(os.path.join(OUT_DIR, base + '.neg'), 'w') as f:
            print('\n'.join(neg), file=f)


    print(f'Saved output to `{OUT_DIR}`.')


if __name__ == '__main__':
    main()
