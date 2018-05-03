# Log

From [Dustin Tran](http://dustintran.com/blog/a-research-to-engineering-workflow):

The directory `log/` records logs for visualizing learning. Each experiment belongs in a subdirectory with the same convention as `checkpoints/`. One benefit of Edward is that for logging, you can simply pass an argument as `inference.initialize(logdir='log/' + subdir)`. Default TensorFlow summaries are tracked which can then be visualized using TensorBoard (more on this next).
