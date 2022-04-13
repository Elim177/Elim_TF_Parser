description: Functional interface for the densely-connected layer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.layers.dense" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.layers.dense

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/legacy_tf_layers/core.py#L144-L255">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Functional interface for the densely-connected layer.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.layers.dense(
    inputs, units, activation=None, use_bias=(True), kernel_initializer=None,
    bias_initializer=tf.compat.v1.zeros_initializer(), kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, trainable=(True), name=None, reuse=None
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
layer is <a href="../../../../tf/keras/layers/Dense.md"><code>tf.keras.layers.Dense</code></a>.


#### Structural Mapping to Native TF2

None of the supported arguments have changed name.

Before:

```python
 y = tf.compat.v1.layers.dense(x, units=3)
```

After:

To migrate code using TF1 functional layers use the [Keras Functional API]
(https://www.tensorflow.org/guide/keras/functional):

```python
 x = tf.keras.Input((28,))
 y = tf.keras.layers.Dense(units=3)(x)
 model = tf.keras.Model(x, y)
```


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

This layer implements the operation:
`outputs = activation(inputs * kernel + bias)`
where `activation` is the activation function passed as the `activation`
argument (if not `None`), `kernel` is a weights matrix created by the layer,
and `bias` is a bias vector created by the layer
(only if `use_bias` is `True`).

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
`units`
</td>
<td>
Integer or Long, dimensionality of the output space.
</td>
</tr><tr>
<td>
`activation`
</td>
<td>
Activation function (callable). Set it to None to maintain a
linear activation.
</td>
</tr><tr>
<td>
`use_bias`
</td>
<td>
Boolean, whether the layer uses a bias.
</td>
</tr><tr>
<td>
`kernel_initializer`
</td>
<td>
Initializer function for the weight matrix.
If `None` (default), weights are initialized using the default
initializer used by <a href="../../../../tf/compat/v1/get_variable.md"><code>tf.compat.v1.get_variable</code></a>.
</td>
</tr><tr>
<td>
`bias_initializer`
</td>
<td>
Initializer function for the bias.
</td>
</tr><tr>
<td>
`kernel_regularizer`
</td>
<td>
Regularizer function for the weight matrix.
</td>
</tr><tr>
<td>
`bias_regularizer`
</td>
<td>
Regularizer function for the bias.
</td>
</tr><tr>
<td>
`activity_regularizer`
</td>
<td>
Regularizer function for the output.
</td>
</tr><tr>
<td>
`kernel_constraint`
</td>
<td>
An optional projection function to be applied to the
kernel after being updated by an `Optimizer` (e.g. used to implement
norm constraints or value constraints for layer weights). The function
must take as input the unprojected variable and must return the
projected variable (which must have the same shape). Constraints are
not safe to use when doing asynchronous distributed training.
</td>
</tr><tr>
<td>
`bias_constraint`
</td>
<td>
An optional projection function to be applied to the
bias after being updated by an `Optimizer`.
</td>
</tr><tr>
<td>
`trainable`
</td>
<td>
Boolean, if `True` also add variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see <a href="../../../../tf/Variable.md"><code>tf.Variable</code></a>).
</td>
</tr><tr>
<td>
`name`
</td>
<td>
String, the name of the layer.
</td>
</tr><tr>
<td>
`reuse`
</td>
<td>
Boolean, whether to reuse the weights of a previous layer
by the same name.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Output tensor the same shape as `inputs` except the last dimension is of
size `units`.
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


