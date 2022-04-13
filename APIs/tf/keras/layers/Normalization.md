description: A preprocessing layer which normalizes continuous features.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.Normalization" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="adapt"/>
<meta itemprop="property" content="compile"/>
<meta itemprop="property" content="reset_state"/>
<meta itemprop="property" content="update_state"/>
</div>

# tf.keras.layers.Normalization

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/preprocessing/normalization.py#L28-L279">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A preprocessing layer which normalizes continuous features.

Inherits From: [`PreprocessingLayer`](../../../tf/keras/layers/experimental/preprocessing/PreprocessingLayer.md), [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.experimental.preprocessing.Normalization`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.layers.Normalization`, `tf.compat.v1.keras.layers.experimental.preprocessing.Normalization`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.Normalization(
    axis=-1, mean=None, variance=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer will shift and scale inputs into a distribution centered around
0 with standard deviation 1. It accomplishes this by precomputing the mean and
variance of the data, and calling `(input - mean) / sqrt(var)` at runtime.

The mean and variance values for the layer must be either supplied on
construction or learned via `adapt()`. `adapt()` will compute the mean and
variance of the data and store them as the layer's weights. `adapt()` should
be called before `fit()`, `evaluate()`, or `predict()`.

For an overview and full list of preprocessing layers, see the preprocessing
[guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`axis`
</td>
<td>
Integer, tuple of integers, or None. The axis or axes that should
have a separate mean and variance for each index in the shape. For
example, if shape is `(None, 5)` and `axis=1`, the layer will track 5
separate mean and variance values for the last axis. If `axis` is set to
`None`, the layer will normalize all elements in the input by a scalar
mean and variance. Defaults to -1, where the last axis of the input is
assumed to be a feature dimension and is normalized per index. Note that
in the specific case of batched scalar inputs where the only axis is the
batch axis, the default will normalize each index in the batch
separately. In this case, consider passing `axis=None`.
</td>
</tr><tr>
<td>
`mean`
</td>
<td>
The mean value(s) to use during normalization. The passed value(s)
will be broadcast to the shape of the kept axes above; if the value(s)
cannot be broadcast, an error will be raised when this layer's `build()`
method is called.
</td>
</tr><tr>
<td>
`variance`
</td>
<td>
The variance value(s) to use during normalization. The passed
value(s) will be broadcast to the shape of the kept axes above; if the
value(s) cannot be broadcast, an error will be raised when this layer's
`build()` method is called.
</td>
</tr>
</table>



#### Examples:



Calculate a global mean and variance by analyzing the dataset in `adapt()`.

```
>>> adapt_data = np.array([1., 2., 3., 4., 5.], dtype='float32')
>>> input_data = np.array([1., 2., 3.], dtype='float32')
>>> layer = tf.keras.layers.Normalization(axis=None)
>>> layer.adapt(adapt_data)
>>> layer(input_data)
<tf.Tensor: shape=(3,), dtype=float32, numpy=
array([-1.4142135, -0.70710677, 0.], dtype=float32)>
```

Calculate a mean and variance for each index on the last axis.

```
>>> adapt_data = np.array([[0., 7., 4.],
...                        [2., 9., 6.],
...                        [0., 7., 4.],
...                        [2., 9., 6.]], dtype='float32')
>>> input_data = np.array([[0., 7., 4.]], dtype='float32')
>>> layer = tf.keras.layers.Normalization(axis=-1)
>>> layer.adapt(adapt_data)
>>> layer(input_data)
<tf.Tensor: shape=(1, 3), dtype=float32, numpy=
array([0., 0., 0.], dtype=float32)>
```

Pass the mean and variance directly.

```
>>> input_data = np.array([[1.], [2.], [3.]], dtype='float32')
>>> layer = tf.keras.layers.Normalization(mean=3., variance=2.)
>>> layer(input_data)
<tf.Tensor: shape=(3, 1), dtype=float32, numpy=
array([[-1.4142135 ],
       [-0.70710677],
       [ 0.        ]], dtype=float32)>
```



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
compiled <a href="../../../tf/Graph.md"><code>tf.Graph</code></a>s that call the layer. This does not affect the layer use
when adapting each layer only once, but if you adapt a layer multiple times
you will need to take care to re-compile any compiled functions as follows:

 * If you are adding a preprocessing layer to a <a href="../../../tf/keras/Model.md"><code>keras.Model</code></a>, you need to
   call `model.compile` after each subsequent call to `adapt`.
 * If you are calling a preprocessing layer inside <a href="../../../tf/data/Dataset.md#map"><code>tf.data.Dataset.map</code></a>,
   you should call `map` again on the input <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> after each
   `adapt`.
 * If you are using a <a href="../../../tf/function.md"><code>tf.function</code></a> directly which calls a preprocessing
   layer, you need to call <a href="../../../tf/function.md"><code>tf.function</code></a> again on your callable after
   each subsequent call to `adapt`.

<a href="../../../tf/keras/Model.md"><code>tf.keras.Model</code></a> example with multiple adapts:

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

<a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> example with multiple adapts:

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
form of datasets, generators, or <a href="../../../tf/keras/utils/Sequence.md"><code>keras.utils.Sequence</code></a> instances
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
<a href="../../../tf/data.md"><code>tf.data</code></a> dataset, and 'steps' is None, the epoch will run until
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
will not be wrapped in a <a href="../../../tf/function.md"><code>tf.function</code></a>. Recommended to leave this as
`None` unless your `Model` cannot be run inside a <a href="../../../tf/function.md"><code>tf.function</code></a>.
steps_per_execution: Int. Defaults to 1. The number of batches to run
  during each <a href="../../../tf/function.md"><code>tf.function</code></a> call. Running multiple batches inside a
  single <a href="../../../tf/function.md"><code>tf.function</code></a> call can greatly improve performance on TPUs or
  small models with a large Python overhead.
</td>
</tr>
</table>



<h3 id="reset_state"><code>reset_state</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/preprocessing/normalization.py#L233-L239">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_state()
</code></pre>

Resets the statistics of the preprocessing layer.


<h3 id="update_state"><code>update_state</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/preprocessing/normalization.py#L196-L231">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_state(
    data
)
</code></pre>

Accumulates statistics for the preprocessing layer.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`data`
</td>
<td>
A mini-batch of inputs to the layer.
</td>
</tr>
</table>





