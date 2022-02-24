description: Object that returns a <a href="../../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> upon invoking.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.experimental.DatasetCreator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.utils.experimental.DatasetCreator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/utils/dataset_creator.py#L22-L110">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Object that returns a <a href="../../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> upon invoking.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.experimental.DatasetCreator(
    dataset_fn, input_options=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

<a href="../../../../tf/keras/utils/experimental/DatasetCreator.md"><code>tf.keras.utils.experimental.DatasetCreator</code></a> is designated as a supported type
for `x`, or the input, in <a href="../../../../tf/keras/Model.md#fit"><code>tf.keras.Model.fit</code></a>. Pass an instance of this class
to `fit` when using a callable (with a `input_context` argument) that returns
a <a href="../../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a>.

```python
model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
model.compile(tf.keras.optimizers.SGD(), loss="mse")

def dataset_fn(input_context):
  global_batch_size = 64
  batch_size = input_context.get_per_replica_batch_size(global_batch_size)
  dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat()
  dataset = dataset.shard(
      input_context.num_input_pipelines, input_context.input_pipeline_id)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(2)
  return dataset

input_options = tf.distribute.InputOptions(
    experimental_fetch_to_device=True,
    experimental_per_replica_buffer_size=2)
model.fit(tf.keras.utils.experimental.DatasetCreator(
    dataset_fn, input_options=input_options), epochs=10, steps_per_epoch=10)
```

<a href="../../../../tf/keras/Model.md#fit"><code>Model.fit</code></a> usage with `DatasetCreator` is intended to work across all
<a href="../../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a>s, as long as <a href="../../../../tf/distribute/MirroredStrategy.md#scope"><code>Strategy.scope</code></a> is used at model
creation:

```python
strategy = tf.distribute.experimental.ParameterServerStrategy(
    cluster_resolver)
with strategy.scope():
  model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
model.compile(tf.keras.optimizers.SGD(), loss="mse")

def dataset_fn(input_context):
  ...

input_options = ...
model.fit(tf.keras.utils.experimental.DatasetCreator(
    dataset_fn, input_options=input_options), epochs=10, steps_per_epoch=10)
```

Note: When using `DatasetCreator`, `steps_per_epoch` argument in <a href="../../../../tf/keras/Model.md#fit"><code>Model.fit</code></a>
must be provided as the cardinality of such input cannot be inferred.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dataset_fn`
</td>
<td>
A callable that takes a single argument of type
<a href="../../../../tf/distribute/InputContext.md"><code>tf.distribute.InputContext</code></a>, which is used for batch size calculation and
cross-worker input pipeline sharding (if neither is needed, the
`InputContext` parameter can be ignored in the `dataset_fn`), and returns
a <a href="../../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a>.
</td>
</tr><tr>
<td>
`input_options`
</td>
<td>
Optional <a href="../../../../tf/distribute/InputOptions.md"><code>tf.distribute.InputOptions</code></a>, used for specific
options when used with distribution, for example, whether to prefetch
dataset elements to accelerator device memory or host device memory, and
prefetch buffer size in the replica device memory. No effect if not used
with distributed training. See <a href="../../../../tf/distribute/InputOptions.md"><code>tf.distribute.InputOptions</code></a> for more
information.
</td>
</tr>
</table>



## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/utils/dataset_creator.py#L102-L110">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    *args, **kwargs
)
</code></pre>

Call self as a function.




