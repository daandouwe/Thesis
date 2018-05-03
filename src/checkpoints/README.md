# Checkpoints

From [Dustin Tran](http://dustintran.com/blog/a-research-to-engineering-workflow):

The folder `checkpoints/` records saved model parameters during training. Use `tf.train.Saver` to save parameters as the algorithm runs every fixed number of iterations. This helps with running long experiments, where you might want to cut the experiment short and later restore the parameters. Each experiment outputs a subdirectory in `checkpoints/` with the convention, `20170524_192314_batch_size_25_lr_1e-4/`. The first number is the date `(YYYYMMDD)`; the second is the timestamp `(%H%M%S)`; and the rest is hyperparameters.
