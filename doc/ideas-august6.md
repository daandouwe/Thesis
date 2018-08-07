# Data
* Use an Item class two wrap indices and tokens together:
  * `Item(word, index)`, then `word = item.word` and `index = item.index`.
  * Use a tokenizer to process input from terminal?

# Embeddings
* Ask Wilker if he knows a nice default character embedding method (e.g. convolutions).
* Elmo embeddings: figure out what the fuss is about, and incorporate them (only discriminative!).
  * Compute embeddings once: save to file, then load like it were glove. But then sentence_id -> token_id -> vector.
* Glove embeddings are done.
* FastText: drop-in.

# Model
* Make prediction 2-step: first predict from `(open_nonterminal, shift, reduce)`, then if `open_nonterminal`, predict `NT(X)`.
* This makes generative model easier: predict from `(open_nonterminal, generate, reduce)` then if `generate` predict word.

# Prediction
* Add beam-search to prediction: see if this improves f1.
* Ancestral sampler decoder.

# Training
Training from https://github.com/nikitakit/self-attentive-parser/blob/master/src/main.py:
```
print("Initializing optimizer...")
 trainable_parameters = [param for param in parser.parameters() if param.requires_grad]
 trainer = torch.optim.Adam(trainable_parameters, lr=1., betas=(0.9, 0.98), eps=1e-9)
 if load_path is not None:
     trainer.load_state_dict(info['trainer'])

 def set_lr(new_lr):
     for param_group in trainer.param_groups:
         param_group['lr'] = new_lr

 assert hparams.step_decay, "Only step_decay schedule is supported"

 warmup_coeff = hparams.learning_rate / hparams.learning_rate_warmup_steps
 scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
     trainer, 'max',
     factor=hparams.step_decay_factor,
     patience=hparams.step_decay_patience,
     verbose=True,
 )
 def schedule_lr(iteration):
     iteration = iteration + 1
     if iteration <= hparams.learning_rate_warmup_steps:
         set_lr(iteration * warmup_coeff)

 clippable_parameters = trainable_parameters
 grad_clip_threshold = np.inf if hparams.clip_grad_norm == 0 else hparams.clip_grad_norm
```
