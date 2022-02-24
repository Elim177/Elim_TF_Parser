description: Enable visualizations for TensorBoard.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.callbacks.TensorBoard" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="set_model"/>
<meta itemprop="property" content="set_params"/>
</div>

# tf.keras.callbacks.TensorBoard

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/callbacks.py#L2033-L2599">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Enable visualizations for TensorBoard.

Inherits From: [`Callback`](../../../tf/keras/callbacks/Callback.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.callbacks.TensorBoard(
    log_dir=&#x27;logs&#x27;, histogram_freq=0, write_graph=(True),
    write_images=(False), write_steps_per_second=(False),
    update_freq=&#x27;epoch&#x27;, profile_batch=0, embeddings_freq=0,
    embeddings_metadata=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

TensorBoard is a visualization tool provided with TensorFlow.

This callback logs events for TensorBoard, including:

* Metrics summary plots
* Training graph visualization
* Activation histograms
* Sampled profiling

When used in <a href="../../../tf/keras/Model.md#evaluate"><code>Model.evaluate</code></a>, in addition to epoch summaries, there will be
a summary that records evaluation metrics vs `Model.optimizer.iterations`
written. The metric names will be prepended with `evaluation`, with
`Model.optimizer.iterations` being the step in the visualized TensorBoard.

If you have installed TensorFlow with pip, you should be able
to launch TensorBoard from the command line:

```
tensorboard --logdir=path_to_your_logs
```

You can find more information about TensorBoard
[here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`log_dir`
</td>
<td>
the path of the directory where to save the log files to be
parsed by TensorBoard. e.g. log_dir = os.path.join(working_dir, 'logs')
This directory should not be reused by any other callbacks.
</td>
</tr><tr>
<td>
`histogram_freq`
</td>
<td>
frequency (in epochs) at which to compute activation and
weight histograms for the layers of the model. If set to 0, histograms
won't be computed. Validation data (or split) must be specified for
histogram visualizations.
</td>
</tr><tr>
<td>
`write_graph`
</td>
<td>
whether to visualize the graph in TensorBoard. The log file
can become quite large when write_graph is set to True.
</td>
</tr><tr>
<td>
`write_images`
</td>
<td>
whether to write model weights to visualize as image in
TensorBoard.
</td>
</tr><tr>
<td>
`write_steps_per_second`
</td>
<td>
whether to log the training steps per second into
Tensorboard. This supports both epoch and batch frequency logging.
</td>
</tr><tr>
<td>
`update_freq`
</td>
<td>
`'batch'` or `'epoch'` or integer. When using `'batch'`,
writes the losses and metrics to TensorBoard after each batch. The same
applies for `'epoch'`. If using an integer, let's say `1000`, the
callback will write the metrics and losses to TensorBoard every 1000
batches. Note that writing too frequently to TensorBoard can slow down
your training.
</td>
</tr><tr>
<td>
`profile_batch`
</td>
<td>
Profile the batch(es) to sample compute characteristics.
profile_batch must be a non-negative integer or a tuple of integers.
A pair of positive integers signify a range of batches to profile.
By default, profiling is disabled.
</td>
</tr><tr>
<td>
`embeddings_freq`
</td>
<td>
frequency (in epochs) at which embedding layers will be
visualized. If set to 0, embeddings won't be visualized.
</td>
</tr><tr>
<td>
`embeddings_metadata`
</td>
<td>
Dictionary which maps embedding layer names to the
filename of a file in which to save metadata for the embedding layer.
In case the same metadata file is to be
used for all embedding layers, a single filename can be passed.
</td>
</tr>
</table>



#### Examples:




#### Basic usage:



```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
# Then run the tensorboard command to view the visualizations.
```

Custom batch-level summaries in a subclassed Model:

```python
class MyModel(tf.keras.Model):

  def build(self, _):
    self.dense = tf.keras.layers.Dense(10)

  def call(self, x):
    outputs = self.dense(x)
    tf.summary.histogram('outputs', outputs)
    return outputs

model = MyModel()
model.compile('sgd', 'mse')

# Make sure to set `update_freq=N` to log a batch-level summary every N batches.
# In addition to any `tf.summary` contained in `Model.call`, metrics added in
# `Model.compile` will be logged every N batches.
tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)
model.fit(x_train, y_train, callbacks=[tb_callback])
```

Custom batch-level summaries in a Functional API Model:

```python
def my_summary(x):
  tf.summary.histogram('x', x)
  return x

inputs = tf.keras.Input(10)
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Lambda(my_summary)(x)
model = tf.keras.Model(inputs, outputs)
model.compile('sgd', 'mse')

# Make sure to set `update_freq=N` to log a batch-level summary every N batches.
# In addition to any `tf.summary` contained in `Model.call`, metrics added in
# `Model.compile` will be logged every N batches.
tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)
model.fit(x_train, y_train, callbacks=[tb_callback])
```

#### Profiling:



```python
# Profile a single batch, e.g. the 5th batch.
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs', profile_batch=5)
model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])

# Profile a range of batches, e.g. from 10 to 20.
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs', profile_batch=(10,20))
model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
```

## Methods

<h3 id="set_model"><code>set_model</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/callbacks.py#L2227-L2245">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_model(
    model
)
</code></pre>

Sets Keras model and writes graph if specified.


<h3 id="set_params"><code>set_params</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/callbacks.py#L642-L643">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_params(
    params
)
</code></pre>






