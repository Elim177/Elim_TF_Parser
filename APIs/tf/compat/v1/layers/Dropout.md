description: Applies Dropout to the input.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.layers.dropout" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.layers.dropout

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/legacy_tf_layers/core.py#L325-L401">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Applies Dropout to the input.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.layers.dropout(
    inputs, rate=0.5, noise_shape=None, seed=None, training=(False), name=None
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
layer is <a href="../../../../tf/keras/layers/Dropout.md"><code>tf.keras.layers.Dropout</code></a>.


#### Structural Mapping to Native TF2

None of the supported arguments have changed name.

Before:

```python
 y = tf.compat.v1.layers.dropout(x)
```

After:

To migrate code using TF1 functional layers use the [Keras Functional API]
(https://www.tensorflow.org/guide/keras/functional):

```python
 x = tf.keras.Input((28, 28, 1))
 y = tf.keras.layers.Dropout()(x)
 model = tf.keras.Model(x, y)
```


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

Dropout consists in randomly setting a fraction `rate` of input units to 0
at each update during training time, which helps prevent overfitting.
The units that are kept are scaled by `1 / (1 - rate)`, so that their
sum is unchanged at training time and inference time.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`inputs`
</td>
<td>
Tensor input.
</td>
</tr><tr>
<td>
`rate`
</td>
<td>
The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out
10% of input units.
</td>
</tr><tr>
<td>
`noise_shape`
</td>
<td>
1D tensor of type `int32` representing the shape of the
binary dropout mask that will be multiplied with the input.
For instance, if your inputs have shape
`(batch_size, timesteps, features)`, and you want the dropout mask
to be the same for all timesteps, you can use
`noise_shape=[batch_size, 1, features]`.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
A Python integer. Used to create random seeds. See
<a href="../../../../tf/compat/v1/set_random_seed.md"><code>tf.compat.v1.set_random_seed</code></a>
for behavior.
</td>
</tr><tr>
<td>
`training`
</td>
<td>
Either a Python boolean, or a TensorFlow boolean scalar tensor
(e.g. a placeholder). Whether to return the output in training mode
(apply dropout) or in inference mode (return the input untouched).
</td>
</tr><tr>
<td>
`name`
</td>
<td>
The name of the layer (string).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Output tensor.
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


