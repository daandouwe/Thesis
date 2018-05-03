import torch
import torch.nn as nn
from torch.autograd import Variable
#
# lstm = nn.LSTM(input_size=10, hidden_size=10,
#                         num_layers=1, batch_first=True, bidirectional=False)
#
# print(dir(lstm))
# print(lstm._parameters.keys())
# print(lstm.all_weights)

tensor = Variable(torch.arange(10).view(2, -1))
print(tensor)
idx = [i for i in range(tensor.size(1) - 1, -1, -1)]
idx = Variable(torch.LongTensor(idx))
tensor = tensor.index_select(1, idx)
print(tensor)
