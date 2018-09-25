# Memory notes

The objects that are increasing are `tensors` and `strings`. There is no increase in `lists`.
```bash
Training...
| step      1/ 2840 | loss 145.884 | lr 1.0e-05 |  3.7 sents/sec | eta 3h27m09s | total mem 1018.655M | added mem 1018.655M | tensors 2,531 | strings 70,923 | lists 104,960 | total 2,405,221
| step      2/ 2840 | loss 114.790 | lr 1.5e-05 |  1.7 sents/sec | eta 7h29m58s | total mem 1100.333M | added mem 81.678M | tensors 4,056 | strings 71,478 | lists 104,960 | total 2,407,377
| step      3/ 2840 | loss 147.614 | lr 2.0e-05 |  1.4 sents/sec | eta 9h02m00s | total mem 1212.776M | added mem 112.443M | tensors 6,219 | strings 72,175 | lists 104,960 | total 2,410,304
| step      4/ 2840 | loss 144.090 | lr 2.5e-05 |  1.3 sents/sec | eta 9h46m03s | total mem 1297.891M | added mem 85.115M | tensors 8,387 | strings 72,853 | lists 104,960 | total 2,413,155
| step      5/ 2840 | loss 199.976 | lr 3.0e-05 |  1.2 sents/sec | eta 10h37m10s | total mem 1446.633M | added mem 148.742M | tensors 11,271 | strings 73,773 | lists 104,960 | total 2,417,012
| step      6/ 2840 | loss  96.746 | lr 3.5e-05 |  1.2 sents/sec | eta 10h54m18s | total mem 1448.006M | added mem 1.372M | tensors 12,455 | strings 74,251 | lists 104,960 | total 2,418,803
| step      7/ 2840 | loss 124.385 | lr 4.0e-05 |  1.1 sents/sec | eta 11h10m05s | total mem 1526.518M | added mem 78.512M | tensors 14,461 | strings 74,836 | lists 104,960 | total 2,421,274
```


# 19 september
The following tensors are increasing:

```
Corpus
vocab size: 25,640
train: 101 sentences
dev: 2,416 sentences
test: 1,346 sentences
Training...
===============================================================================
Counter({torch.Size([1, 102]): 41, torch.Size([1, 100]): 35})
140
===============================================================================
| sent-length 31 | total mem 275.505M | added mem 275.505M | total 502,145 | ints 5,152 | tensors 707 | strings 70,337 | increase 502,145
===============================================================================
Counter({torch.Size([1, 100]): 73,
         torch.Size([1, 102]): 73,
         torch.Size([3, 1, 50]): 26,
         torch.Size([4, 1, 50]): 12,
         torch.Size([2, 1, 50]): 10})
360
===============================================================================
| sent-length 32 | total mem 282.550M | added mem 7.045M | total 502,559 | ints 5,175 | tensors 177 | strings 32 | increase 414
===============================================================================
Counter({torch.Size([1, 100]): 85,
         torch.Size([1, 102]): 85,
         torch.Size([3, 1, 50]): 54,
         torch.Size([2, 1, 50]): 30,
         torch.Size([4, 1, 50]): 18,
         torch.Size([5, 1, 50]): 2,
         torch.Size([6, 1, 50]): 2})
442
===============================================================================
| sent-length 12 | total mem 283.103M | added mem 0.553M | total 502,123 | ints 5,183 | tensors -282 | strings -11 | increase -436
===============================================================================
Counter({torch.Size([1, 100]): 101,
         torch.Size([1, 102]): 101,
         torch.Size([3, 1, 50]): 62,
         torch.Size([2, 1, 50]): 32,
         torch.Size([4, 1, 50]): 20,
         torch.Size([5, 1, 50]): 4,
         torch.Size([6, 1, 50]): 2})
488
===============================================================================
| sent-length 16 | total mem 283.685M | added mem 0.582M | total 502,315 | ints 5,192 | tensors 90 | strings 16 | increase 192
===============================================================================
Counter({torch.Size([1, 100]): 119,
         torch.Size([1, 102]): 119,
         torch.Size([3, 1, 50]): 72,
         torch.Size([2, 1, 50]): 34,
         torch.Size([4, 1, 50]): 26,
         torch.Size([5, 1, 50]): 4,
         torch.Size([6, 1, 50]): 2})
542
```

These are the tensors inside the composition function!

`torch.Size([3, 1, 50])` is the tensor of hidden encodings of the head+children sequence:
```python
hf, _ = self.fwd_rnn(xf)  # (batch, seq, hidden_size//2)
hb, _ = self.bwd_rnn(xb)  # (batch, seq, hidden_size//2)
```

`torch.Size([1, 100])` is the reduced tensor:
```python
reduced = self.encoder.composition(head.embedding, children)
```

`torch.Size([1, 102])` is the encoded tensor:
```python
head.encoding = self.encoder(head.embedding)  # give item new encoding
```
