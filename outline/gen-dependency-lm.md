# Dependency-based language model.

> This looks a lot 
A generative dependency grammar language model that uses a shift-reduce parser to generate sentences with corresponding trees. This can be used for language modeling or as the decoder in an NMT system.

Some observations:
* A generative parser can be used as a syntactic language model. See RNNGs (Dyer et al. 2016) (constituency grammar) and Buys & Blunsom (2015) (dependency).
* Dep-arcs can be thought of as linguistically-grounded self-attention, see "Attention is all you need"
* Dep-arcs can take care of long-range dependencies in a transparent way, see "Frustratingly short attention spans"
* Dep-arcs are *hard* attention. Look into ways of training hard attention models, see "Learning Hard Alignments with Variational Inference".

Some ideas for plugging in latent-variables:
* Treat the heads as latent variables: p(x) = sum_y p(x, y), where y are the trees. This way we could work with  If we let go of modeling strict trees we
* Make the attention attention hard (somewhat alike to above)
can get away with modeling categorical variables over possible heads.
* In the transition-based framework this will not work: we need to model something sequential
* Have some latent switching states: switching between 'semantic' generation (could include some kind of 'topic' structure), and 'syntactic' reduce actions.

As decoder for NMT:
* Using the RNNG as a decoder has been tried, see Eriguchi et al. (2017), "Learning to parse and translate improves NMT", and does something.
* Can we plug a dependency based decoder into the Transformer ("Attention is all you need") model? This could be natural: dependencies could induce structural bias into the attention model on the encoder side. (Maybe a generative dependency language model could improve generated text on the decoder side?)
* The transformer (encoder!) is not autoregressive. Maybe this makes it easier to include stochasticity into the architecture?
