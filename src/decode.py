import torch

class Decoder:
    """Decoder base class."""
    def __init__(self):
        pass

class GreedyDecoder(Decoder):
    """Greedy decoder for RNNG."""
    def __init__(self, model):
        super(Greedy, self).__init__()
        self.model = model

    def __call__(self, sentence):
        self.model.initialize(sentence)
        t = 0
        while not self.model.stack.empty:
            t += 1
            # Compute loss
            stack, buffer, history = self.model.get_encoded_input()
            x = torch.cat((buffer, history, stack), dim=-1)
            action_logits = self.model.action_mlp(x)
            # Get highest scoring valid predictions.
            vals, ids = action_logits.sort(descending=True)
            vals, ids = vals.data.squeeze(0), ids.data.squeeze(0)
            i = 0
            action = Action(self.model.dictionary.i2a[ids[i]], ids[i])
            while not self.model.is_valid_action(action):
                i += 1
                action = Action(self.model.dictionary.i2a[ids[i]], ids[i])
            if action.index == self.model.OPEN:
                nonterminal_logits = self.model.nonterminal_mlp(x)
                vals, ids = nonterminal_logits.sort(descending=True)
                vals, ids = vals.data.squeeze(0), ids.data.squeeze(0)
                action.symbol = Item(self.model.dictionary.i2n[ids[0]], ids[0], nonterminal=True)
            self.model.parse_step(action)
        return self.model.stack.tree.linearize()

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
