import os

import numpy as np


def load_glove(words, dim, dir, logfile=None):
    """Loads all the words from the glove vectors of dimension dim in saved in dir."""

    if dir.startswith('~'):
        dir = os.path.expanduser(dir)

    assert dim in (50, 100, 200, 300), f'invalid dim: {dim}, choose from (50, 100, 200, 300).'
    assert os.path.exists(
            os.path.join(dir, f'glove.6B.{dim}d.txt')
        ), 'glove file not availlable.'

    # Load the glove vectors into a dictionary.
    try:  # fastest way, if gensim is installed
        from gensim.models import KeyedVectors
        path = os.path.join(dir, f'glove.6B.{dim}d.gensim.txt')
        # If gensim version does not yet exist, we make it.
        if not os.path.exists(path):
            make_gensim_compatible(dim)
        glove = KeyedVectors.load_word2vec_format(path, binary=False)
    except ImportError:  # works always
        path = os.path.join(dir, f'glove.6B.{dim}d.txt')
        glove = dict()
        with open(path) as f:
            for line in f:
                line = line.strip().split()
                word, vec = line[0], line[1:]
                # FIXME: Hack! On Lisa: Splitting a line with the
                # name `nguyễn` ended up as `nguy` and `n`
                # Using encoding='utf-8' threw yet another error.
                # But this is an ugly fix!
                i = 2
                while len(vec) > dim:
                    word, vec = ' '.join(line[:i]), line[i:]
                    i += 1
                # end of hack
                vec = np.array([float(val) for val in vec])
                glove[word] = vec

    # Get the glove vector for each word in the dictionary and log words not found.
    vectors = get_vectors(words, glove, dim, logfile=None)

    return vectors


def get_vectors(words, vectordict, dim, logfile=None):
    """Get the vectors for the words in vectordict and return as tensor."""
    vectors = []
    for word in words:
        vec = get_vector(word, vectordict, dim, logfile=None)
        vectors.append(vec)
    return np.vstack(vectors)


def get_vector(word, vectordict, dim, logfile=None):
    """Get the word from the vectordict dictionary.

    Tries alternatives if the word is not in the vectordict dictionary,
    and finally initializes random if even this cannot be done. These words
    are printed to logfile. Default initialization is Normal(0,1).
    """
    try:
        vec = vectordict[word]
    # Word not found.
    except KeyError:
        # If the word is uppercase we try lowercase.
        if word.lower() in vectordict:
            vec = vectordict[word.lower()]
        # If word is can be split using characters like `-` and `&` then
        # we take the average embedding of those splits.
        elif splits(word) is not None: # word can be split into parts
            vec = np.zeros(dim) # accumulator
            parts = splits(word)
            for part in parts:
                if part in vectordict:
                    vec += vectordict[part]
                elif part.lower() in vectordict:
                    vec += vectordict[part.lower()]
                else:
                    pass # implicit zero vector for this part
            vec /= len(parts) # Average of the embeddings
        # Otherwise we assign a random vector.
        else:
            vec = np.random.randn(dim) # Random vector.
            if logfile is not None:
                print(word, file=logfile) # print word to logfile
    return vec


def splits(word, chars=('-', '\/', ',', '.', '&', "'")):
    """Tries to split the word with one of the given characters.

    Returns the longest list of chunks given the characters, and returns None
    if the word cannot split with one of the characters.
    Example:
        `worse-than-expected` returns [`worse`, `then`, `expected`]
        `automobile` -> None
    """
    words = None
    longest = 1
    for char in chars:
        chunks = word.split(char)
        if len(chunks) > longest:
            words = chunks
            longest = len(chunks)
    return words


def make_gensim_compatible(dim):
    """Make a glove vector path gensim compatible.

    Prints the number of words and dimension at the top of a copy of the glove
    file with dimension dim.
    """
    in_path  = f'glove.6B.{dim}d.txt'
    out_path = f'glove.6B.{dim}d.gensim.txt'
    length = sum(1 for _ in open(in_path))
    with open(in_path) as f:
        with open(out_path, 'w') as g:
            print(length, dim, file=g)
            print(f.read(), file=g, end='')
