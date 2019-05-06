# Data

## A note on tokenization
There is a difference in tokenization between the PTB and the one-bilion-words corpus. The PTB chooses
```
do n't  are n't  is n't
```
The OBW corpus chooses
```
don 't  aren 't  isn 't
```

We can choose to change this with a regex that maps `n 't` to ` n't`.
