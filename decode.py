import torch

class Decoder:
    """Decoder base class."""
    def __init__(self):
        pass

class GreedyDecoder(Decoder):
    """Greedy decoder for RNNG."""
    def __init__(self):
        super(Greedy, self).__init__()
        pass

class AncestralSamplingDecoder(Decoder):
    """Ancestral sampling decoder for RNNG."""
    def __init__(self):
        super(Greedy, self).__init__()
        pass

class BeamSearchDecoder(Decoder):
    """Beam search decoder for RNNG."""
    def __init__(self):
        super(BeamSearchDecoder, self).__init__()
        pass
