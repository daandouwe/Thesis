from data import wrap


class LossCompute:
    def __init__(self, criterion, device):
        self.criterion = criterion()
        self.device = device

    def __call__(self, logits, y):
        """
        logits (tensor): model predictions.
        y (int): the correct index.
        """
        assert isinstance(y, int), y
        y = wrap([y], self.device)
        return self.criterion(logits, y)


# More loss functions...
