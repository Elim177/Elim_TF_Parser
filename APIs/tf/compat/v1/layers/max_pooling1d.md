description: Max Pooling layer for 1D inputs.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.layers.max_pooling1d" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.layers.max_pooling1d

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/legacy_tf_layers/pooling.py#L224-L294">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Max Pooling layer for 1D inputs.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.layers.max_pooling1d(
    inputs, pool_size, strides, padding=&#x27;valid&#x27;,
    data_format=&#x27;channels_last&#x27;, name=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

This API is not compatible with eager execution or <a href="../../../../tf/function.md"><code>tf.function</code></a>.

Please refer to [tf.layers section of the migration guide]
(https://www.tensorflow.org/guide/migrate#models_based_on_tflayers)
to migrate a TensorFlow v1 model to Keras. The corresponding TensorFlow v2
layer is <a href="../../../../tf/keras/layers/MaxPool1D.md"><code>tf.keras.layers.MaxPooling1D</code></a>.


#### Structural Mapping to Native TF2

None of the supported arguments have changed name.

Before:

```python
 y = tf.compat.v1.layers.max_pooling1d(x, pool_size=2, strides=2)
```

After:

To migrate code using TF1 functional layers use the [Keras Functional API]
(https://www.tensorflow.org/guide/keras/functional):

```python
 x = tf.keras.Input((28, 28, 1))
 y = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(x)
 model = tf.keras.Model(x, y)
```


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`inputs`
</td>
<td>
The tensor over which to pool. Must have rank 3.
</td>
</tr><tr>
<td>
`pool_size`
</td>
<td>
An integer or tuple/list of a single integer,
representing the size of the pooling window.
</td>
</tr><tr>
<td>
`strides`
</td>
<td>
An integer or tuple/list of a single integer, specifying the
strides of the pooling operation.
</td>
</tr><tr>
<td>
`padding`
</td>
<td>
A string. The padding method, either 'valid' or 'same'.
Case-insensitive.
</td>
</tr><tr>
<td>
`data_format`
</td>
<td>
A string, one of `channels_last` (default) or `channels_first`.
The ordering of the dimensions in the inputs.
`channels_last` corresponds to inputs with shape
`(batch, length, channels)` while `channels_first` corresponds to
inputs with shape `(batch, channels, length)`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A string, the name of the layer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The output tensor, of rank 3.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if eager execution is enabled.
</td>
</tr>
</table>


