# A dependency based grammar VAE

This combines [Grammar VAE](Grammar Variational Autoencoder) with RNNGs.

"Generating structures from a continuous space"



1. Basis: A generative dependency based language model (shift-reduce)
2. VAE: Use this to encode sentence+tree into R^n (following Grammar VAE)
3. Decode to give sentences+trees
4. Find a way to disentangle the trees (structure) from content (the words).

The 'generate' action (generative shift) is the content component. This could
include a 'topic model'-like component.
The reduce actions are the syntactic component.


# Latent trees

There is this literature on learning latent trees to encode sentences with, used for discriminative tasks like NLI. This strand of research is reviewed in Bowman et al. (2018) "Do latent tree models identify meaningful structure in language" (the answer btw is pretty much: no).
