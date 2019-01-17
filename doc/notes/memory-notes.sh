# Problem
Run the following with `model_test.py`:
`./main.py train --batch-size 1`
After 1300 updates the memory consumption is around 8 GB.
With `model.py` memory consumption stays at 2.8 GB.
WTF!!!! NOTHING I TRY FIXES IT???????

# Findings

##
```
UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters().
```
## Which object is increasing?

I tried the following during the training script:
```python
all_objects = muppy.get_objects()
tensors = muppy.filter(all_objects, Type=torch.Tensor)
lists = muppy.filter(all_objects, Type=list)
word = muppy.filter(all_objects, Type=Word)
nt = muppy.filter(all_objects, Type=Nonterminal)
action = muppy.filter(all_objects, Type=Action)
leaf = muppy.filter(all_objects, Type=LeafNode)
internal = muppy.filter(all_objects, Type=InternalNode)

print(f'objects {len(all_objects):,}')
print(f'tensors {len(tensors):,}')
print(f'list {len(lists):,}')
print(f'word {len(word):,}')
print(f'nt {len(nt):,}')
print(f'action {len(action):,}')
print(f'leaf {len(leaf):,}')
print(f'internal {len(internal):,}')
```

This resulted in:
```
-------------------------------------------------------
Before minibatch:
objects 2,449,304
list 132,874
tensors 5,522
word 1,173,766
nt 1
action 915,936
leaf 0
internal 2

After minibatch:
objects 2,449,724
list 132,874
tensors 5,577
word 1,173,766
nt 1
action 915,936
leaf 0
internal 2

After update:
objects 2,449,724
list 132,874
tensors 5,577
word 1,173,766
nt 1
action 915,936
leaf 0
internal 2
```
And some steps later:
```
-------------------------------------------------------
Before minibatch:
objects 2,457,507
list 132,874
tensors 9,062
word 1,173,766
nt 1
action 915,936
leaf 0
internal 2

After minibatch:
objects 2,460,623
list 132,874
tensors 10,433
word 1,173,766
nt 1
action 915,936
leaf 0
internal 2

After update:
objects 2,460,623
list 132,874
tensors 10,433
word 1,173,766
nt 1
action 915,936
leaf 0
internal 2
```
Nothing increased except `Tensors`, from 5,522 to 10,433, in a couple of steps.


## Plan

Figure out where exactly these tensors are made. I tried this:

### With Adam optimizer:
```
===============================================================================
Before initialize parser:
objects 2,396,398
tensors 66
not tensors 2,396,332

After initialize parser:
objects 2,396,548
tensors 129
not tensors 2,396,419


===============================================================================
Before initialize parser:
objects 2,397,240
tensors 535
not tensors 2,396,705

After initialize parser:
objects 2,397,135
tensors 421
not tensors 2,396,714


===============================================================================
Before initialize parser:
objects 2,398,307
tensors 1,081
not tensors 2,397,226

After initialize parser:
objects 2,397,702
tensors 655
not tensors 2,397,047


===============================================================================
Before initialize parser:
objects 2,398,520
tensors 1,115
not tensors 2,397,405

After initialize parser:
objects 2,398,047
tensors 795
not tensors 2,397,252


===============================================================================
Before initialize parser:
objects 2,398,496
tensors 1,042
not tensors 2,397,454

After initialize parser:
objects 2,398,257
tensors 881
not tensors 2,397,376


===============================================================================
Before initialize parser:
objects 2,398,550
tensors 1,042
not tensors 2,397,508

After initialize parser:
objects 2,398,499
tensors 985
not tensors 2,397,514


===============================================================================
Before initialize parser:
objects 2,399,255
tensors 1,406
not tensors 2,397,849

After initialize parser:
objects 2,398,874
tensors 1,143
not tensors 2,397,731


===============================================================================
Before initialize parser:
objects 2,399,557
tensors 1,522
not tensors 2,398,035

After initialize parser:
objects 2,399,189
tensors 1,273
not tensors 2,397,916
```

### With SGD optimizer:
```===============================================================================
Before initialize parser:
objects 2,396,397
tensors 66
not tensors 2,396,331

After initialize parser:
objects 2,396,547
tensors 129
not tensors 2,396,418


===============================================================================
Before initialize parser:
objects 2,397,077
tensors 427
not tensors 2,396,650

After initialize parser:
objects 2,396,972
tensors 313
not tensors 2,396,659


===============================================================================
Before initialize parser:
objects 2,398,144
tensors 973
not tensors 2,397,171

After initialize parser:
objects 2,397,539
tensors 547
not tensors 2,396,992


===============================================================================
Before initialize parser:
objects 2,398,357
tensors 1,007
not tensors 2,397,350

After initialize parser:
objects 2,397,884
tensors 687
not tensors 2,397,197


===============================================================================
Before initialize parser:
objects 2,398,333
tensors 934
not tensors 2,397,399

After initialize parser:
objects 2,398,094
tensors 773
not tensors 2,397,321


===============================================================================
Before initialize parser:
objects 2,398,387
tensors 934
not tensors 2,397,453

After initialize parser:
objects 2,398,336
tensors 877
not tensors 2,397,459


===============================================================================
Before initialize parser:
objects 2,399,092
tensors 1,298
not tensors 2,397,794

After initialize parser:
objects 2,398,711
tensors 1,035
not tensors 2,397,676


===============================================================================
Before initialize parser:
objects 2,399,394
tensors 1,414
not tensors 2,397,980

After initialize parser:
objects 2,399,026
tensors 1,165
not tensors 2,397,861


===============================================================================
Before initialize parser:
objects 2,399,460
tensors 1,405
not tensors 2,398,055

After initialize parser:
objects 2,399,221
tensors 1,245
not tensors 2,397,976


===============================================================================
Before initialize parser:
objects 2,399,550
tensors 1,424
not tensors 2,398,126

After initialize parser:
objects 2,399,474
tensors 1,353
not tensors 2,398,121

| step     10/45446 | loss 158.227 | lr 5.5e-05 |  0.0 sents/sec | eta 12d10h40m10s

===============================================================================
Before initialize parser:
objects 2,400,386
tensors 1,854
not tensors 2,398,532

After initialize parser:
objects 2,399,920
tensors 1,537
not tensors 2,398,383
```

### With shit loads of del statements everywhere:
```
===============================================================================
Before initialize parser:
objects 2,396,398
tensors 66
not tensors 2,396,332

After initialize parser:
objects 2,396,548
tensors 129
not tensors 2,396,419


===============================================================================
Before initialize parser:
objects 2,397,240
tensors 535
not tensors 2,396,705

After initialize parser:
objects 2,397,135
tensors 421
not tensors 2,396,714


===============================================================================
Before initialize parser:
objects 2,398,307
tensors 1,081
not tensors 2,397,226

After initialize parser:
objects 2,397,702
tensors 655
not tensors 2,397,047


===============================================================================
Before initialize parser:
objects 2,398,520
tensors 1,115
not tensors 2,397,405

After initialize parser:
objects 2,398,047
tensors 795
not tensors 2,397,252


===============================================================================
Before initialize parser:
objects 2,398,496
tensors 1,042
not tensors 2,397,454

After initialize parser:
objects 2,398,257
tensors 881
not tensors 2,397,376


===============================================================================
Before initialize parser:
objects 2,398,550
tensors 1,042
not tensors 2,397,508

After initialize parser:
objects 2,398,499
tensors 985
not tensors 2,397,514


===============================================================================
Before initialize parser:
objects 2,399,255
tensors 1,406
not tensors 2,397,849

After initialize parser:
objects 2,398,874
tensors 1,143
not tensors 2,397,731


===============================================================================
Before initialize parser:
objects 2,399,557
tensors 1,522
not tensors 2,398,035

After initialize parser:
objects 2,399,189
tensors 1,273
not tensors 2,397,916


===============================================================================
Before initialize parser:
objects 2,399,623
tensors 1,513
not tensors 2,398,110

After initialize parser:
objects 2,399,384
tensors 1,353
not tensors 2,398,031


===============================================================================
Before initialize parser:
objects 2,399,713
tensors 1,532
not tensors 2,398,181

After initialize parser:
objects 2,399,637
tensors 1,461
not tensors 2,398,176

| step     10/45446 | loss 158.267 | lr 5.5e-05 |  0.0 sents/sec | eta 13d4h28m15s

===============================================================================
Before initialize parser:
objects 2,400,549
tensors 1,962
not tensors 2,398,587

After initialize parser:
objects 2,400,083
tensors 1,645
not tensors 2,398,438


===============================================================================
Before initialize parser:
objects 2,400,622
tensors 1,946
not tensors 2,398,676

After initialize parser:
objects 2,400,395
tensors 1,777
not tensors 2,398,618


===============================================================================
Before initialize parser:
objects 2,401,075
tensors 2,157
not tensors 2,398,918

After initialize parser:
objects 2,400,690
tensors 1,899
not tensors 2,398,791


===============================================================================
Before initialize parser:
objects 2,401,154
tensors 2,153
not tensors 2,399,001

After initialize parser:
objects 2,400,932
tensors 2,001
not tensors 2,398,931


===============================================================================
Before initialize parser:
objects 2,401,324
tensors 2,219
not tensors 2,399,105

After initialize parser:
objects 2,401,117
tensors 2,079
not tensors 2,399,038


===============================================================================
Before initialize parser:
objects 2,401,464
tensors 2,270
not tensors 2,399,194

After initialize parser:
objects 2,401,344
tensors 2,175
not tensors 2,399,169


===============================================================================
Before initialize parser:
objects 2,401,988
tensors 2,531
not tensors 2,399,457

After initialize parser:
objects 2,401,732
tensors 2,341
not tensors 2,399,391
```
