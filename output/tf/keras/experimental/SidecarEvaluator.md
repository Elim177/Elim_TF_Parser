description: A class designed for a dedicated evaluator task.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.experimental.SidecarEvaluator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="start"/>
</div>

# tf.keras.experimental.SidecarEvaluator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/distribute/sidecar_evaluator.py#L46-L245">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A class designed for a dedicated evaluator task.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.experimental.SidecarEvaluator(
    model, data, checkpoint_dir, steps=None, max_evaluations=None, callbacks=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

`SidecarEvaluator` is expected to be run in a process on a separate machine
from the training cluster. It is meant for the purpose of a dedicated
evaluator, evaluating the metric results of a training cluster which has one
or more workers performing the training, and saving checkpoints.

The `SidecarEvaluator` API is compatible with both Custom Training Loop (CTL),
and Keras <a href="../../../tf/keras/Model.md#fit"><code>Model.fit</code></a> to be used in the training cluster. Using the model
(with compiled metrics) provided at `__init__`, `SidecarEvaluator` repeatedly
performs evaluation "epochs" when it finds a checkpoint that has not yet been
used. Depending on the `steps` argument, an eval epoch is evaluation over all
eval data, or up to certain number of steps (batches). See examples below for
how the training program should save the checkpoints in order to be recognized
by `SidecarEvaluator`.

Since under the hood, `SidecarEvaluator` uses `model.evaluate` for evaluation,
it also supports arbitrary Keras callbacks. That is, if one or more callbacks
are provided, their `on_test_batch_begin` and `on_test_batch_end` methods are
called at the start and end of a batch, and their `on_test_begin` and
`on_test_end` are called at the start and end of an evaluation epoch. Note
that `SidecarEvaluator` may skip some checkpoints because it always picks up
the latest checkpoint available, and during an evaluation epoch, multiple
checkpoints can be produced from the training side.

#### Example:


```python
model = tf.keras.models.Sequential(...)
model.compile(metrics=tf.keras.metrics.SparseCategoricalAccuracy(
    name="eval_metrics"))
data = tf.data.Dataset.from_tensor_slices(...)

tf.keras.experimental.SidecarEvaluator(
    model=model,
    data=data,
    checkpoint_dir='/tmp/checkpoint_dir',  # dir for training-saved checkpoint
    steps=None,  # Eval until dataset is exhausted
    max_evaluations=None,  # The evaluation needs to be stopped manually
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir='/tmp/log_dir')]
).start()
```

<a href="../../../tf/keras/experimental/SidecarEvaluator.md#start"><code>SidecarEvaluator.start</code></a> writes a series of summary
files which can be visualized by tensorboard (which provides a webpage link):

```bash
$ tensorboard --logdir=/tmp/log_dir/validation
...
TensorBoard 2.4.0a0 at http://host:port (Press CTRL+C to quit)
```

If the training cluster uses a CTL, the `checkpoint_dir` should contain
checkpoints that track both `model` and `optimizer`, to fulfill
`SidecarEvaluator`'s expectation. This can be done by a
<a href="../../../tf/train/Checkpoint.md"><code>tf.train.Checkpoint</code></a> and a <a href="../../../tf/train/CheckpointManager.md"><code>tf.train.CheckpointManager</code></a>:

```python
checkpoint_dir = ...  # Same `checkpoint_dir` supplied to `SidecarEvaluator`.
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, checkpoint_dir=..., max_to_keep=...)
checkpoint_manager.save()
```

If the training cluster uses Keras <a href="../../../tf/keras/Model.md#fit"><code>Model.fit</code></a> API, a
<a href="../../../tf/keras/callbacks/ModelCheckpoint.md"><code>tf.keras.callbacks.ModelCheckpoint</code></a> should be used, with
`save_weights_only=True`, and the `filepath` should have 'ckpt-{epoch}'
appended:

```python
checkpoint_dir = ...  # Same `checkpoint_dir` supplied to `SidecarEvaluator`.
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'ckpt-{epoch}'),
    save_weights_only=True)
model.fit(dataset, epochs, callbacks=[model_checkpoint])
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model`
</td>
<td>
Model to use for evaluation. The model object used here should be a
<a href="../../../tf/keras/Model.md"><code>tf.keras.Model</code></a>, and should be the same as the one that is used in
training, where <a href="../../../tf/keras/Model.md"><code>tf.keras.Model</code></a>s are checkpointed. The model should
have one or more metrics compiled before using `SidecarEvaluator`.
</td>
</tr><tr>
<td>
`data`
</td>
<td>
The input data for evaluation. `SidecarEvaluator` supports all data
types that Keras `model.evaluate` supports as the input data `x`, such
as a <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a>.
</td>
</tr><tr>
<td>
`checkpoint_dir`
</td>
<td>
Directory where checkpoint files are saved.
</td>
</tr><tr>
<td>
`steps`
</td>
<td>
Number of steps to perform evaluation for, when evaluating a single
checkpoint file. If `None`, evaluation continues until the dataset is
exhausted. For repeated evaluation dataset, user must specify `steps` to
avoid infinite evaluation loop.
</td>
</tr><tr>
<td>
`max_evaluations`
</td>
<td>
Maximum number of the checkpoint file to be evaluated,
for `SidecarEvaluator` to know when to stop. The evaluator will stop
after it evaluates a checkpoint filepath ending with
'<ckpt_name>-<max_evaluations>'. If using
<a href="../../../tf/train/CheckpointManager.md#save"><code>tf.train.CheckpointManager.save</code></a> for saving checkpoints, the kth saved
checkpoint has the filepath suffix '<ckpt_name>-<k>' (k=1 for the first
saved), and if checkpoints are saved every epoch after training, the
filepath saved at the kth epoch would end with '<ckpt_name>-<k>. Thus,
if training runs for n epochs, and the evaluator should end after the
training finishes, use n for this parameter. Note that this is not
necessarily equal to the number of total evaluations, since some
checkpoints may be skipped if evaluation is slower than checkpoint
creation. If `None`, `SidecarEvaluator` will evaluate indefinitely, and
the user must terminate evaluator program themselves.
</td>
</tr><tr>
<td>
`callbacks`
</td>
<td>
List of <a href="../../../tf/keras/callbacks/Callback.md"><code>keras.callbacks.Callback</code></a> instances to apply during
evaluation. See [callbacks](/api_docs/python/tf/keras/callbacks).
</td>
</tr>
</table>



## Methods

<h3 id="start"><code>start</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/distribute/sidecar_evaluator.py#L176-L245">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>start()
</code></pre>

Starts the evaluation loop.




