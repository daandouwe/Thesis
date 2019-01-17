# RNNG training notes
I peeked into the actual Dyer RNNG c++ code. This is what I learned

## Learning rate
The RNNG by dyer uses learning rate decay! Evidence in `cnn/cnn/training.h`:
```cpp
struct Trainer {
  explicit Trainer(Model* m, real lam, real e0) :
    eta0(e0), eta(e0), eta_decay(), epoch(), lambda(lam), clipping_enabled(true), clip_threshold(5), clips(), updates(), model(m) {}
  virtual ~Trainer();

  virtual void update(real scale = 1.0) = 0;
  void update_epoch(real r = 1) {
    epoch += r;
    eta = eta0 / (1 + epoch * eta_decay);
  }

  struct SimpleSGDTrainer : public Trainer {
  explicit SimpleSGDTrainer(Model* m, real lam = 1e-6, real e0 = 0.1) : Trainer(m, lam, e0) {}
  void update(real scale) override;
  void update(const std::vector<LookupParameters*> &lookup_params, const std::vector<Parameters*> &params, real scale = 1);
};
```
And then in `nt-parser/nt-parser.cc`:
```cpp
SimpleSGDTrainer sgd(&model);
//AdamTrainer sgd(&model);
//MomentumSGDTrainer sgd(&model);
//sgd.eta_decay = 0.08;
sgd.eta_decay = 0.05;

...

for (unsigned sii = 0; sii < status_every_i_iterations; ++sii)
     if (si == corpus.sents.size()) {
       si = 0;
       if (first) { first = false; } else { sgd.update_epoch(); }
       cerr << "SHUFFLE\n";
       random_shuffle(order.begin(), order.end());
     }
```
So every epoch the learning rate is scheduled

## Weight decay
The rnng uses w
```cpp
real lambda; // weight regularization (l2)

struct SimpleSGDTrainer : public Trainer {
  explicit SimpleSGDTrainer(Model* m, real lam = 1e-6, real e0 = 0.1) : Trainer(m, lam, e0) {}
  void update(real scale) override;
  void update(const std::vector<LookupParameters*> &lookup_params, const std::vector<Parameters*> &params, real scale = 1);
};
```

## Speed
It claims to go at around 100ms per instance.
```
[epoch=0 eta=0.1 clips=85 updates=100] update #122 (epoch 0.270657) per-action-ppl: 1.18654 per-input-ppl: 1.55083 per-sent-ppl: 19824.2 err: 0.0506482 [96.57ms per instance]
[epoch=0 eta=0.1 clips=95 updates=100] update #123 (epoch 0.272857) per-action-ppl: 1.20679 per-input-ppl: 1.62206 per-sent-ppl: 112755 err: 0.0542899 [100.15ms per instance]
[epoch=0 eta=0.1 clips=93 updates=100] update #124 (epoch 0.275058) per-action-ppl: 1.18209 per-input-ppl: 1.53777 per-sent-ppl: 36631.7 err: 0.0501433 [109.96ms per instance]
[epoch=0 eta=0.1 clips=87 updates=100] update #125 (epoch 0.277258) per-action-ppl: 1.16815 per-input-ppl: 1.49764 per-sent-ppl: 8257.4 err: 0.0470446 [106.79ms per instance]
```
- This means 12500 updates * 0.1s / 60s = 21 minutes for 12500 updates.
- My implementation goes at about 2 instances per second or 500ms per instance. So mine is 5 times slower.
- This means one epoch takes 0.1 * 45000 / 60 = 75 mins
- This means it takes three days (this much time till convergence, they claim) 3 * 24 * 60 = 4320 minutes / 75 = 58 epochs of training!
- WTF.

## Accuracy
After 12500 updates (sentences), the accuracy is 81 Fscore:
```
[epoch=0 eta=0.1 clips=87 updates=100] update #125 (epoch 0.277258) per-action-ppl: 1.16815 per-input-ppl: 1.49764 per-sent-ppl: 8257.4 err: 0.0470446 [106.79ms per instance]
Dev output in /tmp/parser_dev_eval.56495.txt
  **dev (iter=125 epoch=0.277258)	llh=22037.7 ppl: 1.47518 f1: 81.02 err: 0.045822	[2416 sents in 151413 ms]
  new best...writing model to ntparse_pos_0_2_32_128_16_128-pid56495.params ...
[epoch=0 eta=0.1 clips=93 updates=100] update #126 (epoch 0.279459) per-action-ppl: 1.17651 per-input-ppl: 1.52423 per-sent-ppl: 25044.9 err: 0.0503932 [102.71ms per instance]
[epoch=0 eta=0.1 clips=90 updates=100] update #127 (epoch 0.281659) per-action-ppl: 1.18464 per-input-ppl: 1.54847 per-sent-ppl: 47151.9 err: 0.0546371 [109.07ms per instance]
```
