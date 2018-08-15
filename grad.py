import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1, 4))
        print('a:', self.a)

    def forward(self):
        b = self.a**2
        c = b*2
        d = c.mean()
        e = c.sum()
        return d, e


model = Model()
print('parameters:', list(model.parameters()))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print()
for _ in range(3):
    d, e = model()
    print('grad a before:', model.a.grad)
    optimizer.zero_grad()
    print('grad a zero grad:', model.a.grad)
    d.backward(retain_graph=True)
    print('grad a d-backward:', model.a.grad)
    e.backward()
    print('grad a e-backward:', model.a.grad)
    optimizer.step()
    print('grad a step:', model.a.grad)
    print()
