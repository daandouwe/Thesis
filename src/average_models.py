import torch

with open('lisa-experiments/disc-models-3/adam_lr0.001_batch_size32/model.pt', 'rb') as f:
    state1 = torch.load(f, map_location='cpu')
    model1 = state1['model']

print(type(list(model1.named_parameters())[0]))
print(type(list(model1.parameters())[0]))

# with open('lisa-experiments/disc-models-3/adam_lr0.0005_batch_size32/model.pt', 'rb') as f:
#     state2 = torch.load(f, map_location='cpu')
#     model2 = state2['model']
#
#
# params1 = list(model1.named_parameters())
# params2 = list(model2.named_parameters())
#
# for model, params in zip((model1, model2), (params1, params2)):
#     params.append(('buffer.embedding.weight', model.stack.word_embedding.weight))
#     params.append(('history.word_embedding.weight', model.stack.word_embedding.weight))
#     params.append(('history.nt_embedding.weight', model.stack.nt_embedding.weight))
#
# print(params1)
# dict_params = dict(params2)
#
# for name1, param1 in params1:
#     assert name1 in dict_params, name1
#     param2 = dict_params[name1]
#     dict_params[name1].data = ((param1.data + param2.data) / 2)
#
# model1.load_state_dict(dict_params)
# state1['model'] = model1
#
# with open('lisa-experiments/averaged-model.pt', 'wb') as f:
#     torch.save(state, f)
