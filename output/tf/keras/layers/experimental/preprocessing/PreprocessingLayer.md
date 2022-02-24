description: Base class for Preprocessing Layers.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.experimental.preprocessing.PreprocessingLayer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="adapt"/>
<meta itemprop="property" content="compile"/>
</div>

# tf.keras.layers.experimental.preprocessing.PreprocessingLayer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/engine/base_preprocessing_layer.py#L34-L281">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Base class for Preprocessing Layers.

Inherits From: [`Layer`](../../../../../tf/keras/layers/Layer.md), [`Module`](../../../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.layers.experimental.preprocessing.PreprocessingLayer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.experimental.preprocessing.PreprocessingLayer(
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

**Don't use this class directly: it's an abstract base class!** You may
be looking for one of the many built-in
[preprocessing layers](https://keras.io/guides/preprocessing_layers/)
instead.

Preprocessing layers are layers whose state gets computed before model
training starts. They do not get updated during training.
Most preprocessing layers implement an `adapt()` method for state computation.

The `PreprocessingLayer` class is the base class you would subclass to
implement your own preprocessing layers.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`is_adapted`
</td>
<td>
Whether the layer has been fit to data already.
</td>
</tr>
</table>



## Methods

<h3 id="adapt"><code>adapt</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/engine/base_preprocessing_layer.py#L156-L248">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>adapt(
    data, batch_size=None, steps=None
)
</code></pre>

Fits the state of the preprocessing layer to the data being passed.

After calling `adapt` on a layer, a preprocessing layer's state will not
update during training. In order to make preprocessing layers efficient in
any distribution context, they are kept constant with respect to any
compiled <a href="../../../../../tf/Graph.md"><code>tf.Graph</code></a>s that call the layer. This does not affect the layer use
when adapting each layer only once, but if you adapt a layer multiple times
you will need to take care to re-compile any compiled functions as follows:

 * If you are adding a preprocessing layer to a <a href="../../../../../tf/keras/Model.md"><code>keras.Model</code></a>, you need to
   call `model.compile` after each subsequent call to `adapt`.
 * If you are calling a preprocessing layer inside <a href="../../../../../tf/data/Dataset.md#map"><code>tf.data.Dataset.map</code></a>,
   you should call `map` again on the input <a href="../../../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> after each
   `adapt`.
 * If you are using a <a href="../../../../../tf/function.md"><code>tf.function</code></a> directly which calls a preprocessing
   layer, you need to call <a href="../../../../../tf/function.md"><code>tf.function</code></a> again on your callable after
   each subsequent call to `adapt`.

<a href="../../../../../tf/keras/Model.md"><code>tf.keras.Model</code></a> example with multiple adapts:

```
>>> layer = tf.keras.layers.experimental.preprocessing.Normalization(
...     axis=None)
>>> layer.adapt([0, 2])
>>> model = tf.keras.Sequential(layer)
>>> model.predict([0, 1, 2])
array([-1.,  0.,  1.], dtype=float32)
>>> layer.adapt([-1, 1])
>>> model.compile() # This is needed to re-compile model.predict!
>>> model.predict([0, 1, 2])
array([0., 1., 2.], dtype=float32)
```

<a href="../../../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> example with multiple adapts:

```
>>> layer = tf.keras.layers.experimental.preprocessing.Normalization(
...     axis=None)
>>> layer.adapt([0, 2])
>>> input_ds = tf.data.Dataset.range(3)
>>> normalized_ds = input_ds.map(layer)
>>> list(normalized_ds.as_numpy_iterator())
[array([-1.], dtype=float32),
 array([0.], dtype=float32),
 array([1.], dtype=float32)]
>>> layer.adapt([-1, 1])
>>> normalized_ds = input_ds.map(layer) # Re-map over the input dataset.
>>> list(normalized_ds.as_numpy_iterator())
[array([0.], dtype=float32),
 array([1.], dtype=float32),
 array([2.], dtype=float32)]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`data`
</td>
<td>
The data to train on. It can be passed either as a tf.data
Dataset, or as a numpy array.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
Integer or `None`.
Number of samples per state update.
If unspecified, `batch_size` will default to 32.
Do not specify the `batch_size` if your data is in the
form of datasets, generators, or <a href="../../../../../tf/keras/utils/Sequence.md"><code>keras.utils.Sequence</code></a> instances
(since they generate batches).
</td>
</tr><tr>
<td>
`steps`
</td>
<td>
Integer or `None`.
Total number of steps (batches of samples)
When training with input tensors such as
TensorFlow data tensors, the default `None` is equal to
the number of samples in your dataset divided by
the batch size, or 1 if that cannot be determined. If x is a
<a href="../../../../../tf/data.md"><code>tf.data</code></a> dataset, and 'steps' is None, the epoch will run until
the input dataset is exhausted. When passing an infinitely
repeating dataset, you must specify the `steps` argument. This
argument is not supported with array inputs.
</td>
</tr>
</table>



<h3 id="compile"><code>compile</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/engine/base_preprocessing_layer.py#L134-L154">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compile(
    run_eagerly=None, steps_per_execution=None
)
</code></pre>

Configures the layer for `adapt`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`run_eagerly`
</td>
<td>
Bool. Defaults to `False`. If `True`, this `Model`'s logic
will not be wrapped in a <a href="../../../../../tf/function.md"><code>tf.function</code></a>. Recommended to leave this as
`None` unless your `Model` cannot be run inside a <a href="../../../../../tf/function.md"><code>tf.function</code></a>.
steps_per_execution: Int. Defaults to 1. The number of batches to run
  during each <a href="../../../../../tf/function.md"><code>tf.function</code></a> call. Running multiple batches inside a
  single <a href="../../../../../tf/function.md"><code>tf.function</code></a> call can greatly improve performance on TPUs or
  small models with a large Python overhead.
</td>
</tr>
</table>





