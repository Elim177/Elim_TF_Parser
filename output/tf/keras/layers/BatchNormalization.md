description: Layer that normalizes its inputs.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.BatchNormalization" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf.keras.layers.BatchNormalization

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/normalization/batch_normalization.py#L1124-L1265">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Layer that normalizes its inputs.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.BatchNormalization(
    axis=-1, momentum=0.99, epsilon=0.001, center=(True), scale=(True),
    beta_initializer=&#x27;zeros&#x27;, gamma_initializer=&#x27;ones&#x27;,
    moving_mean_initializer=&#x27;zeros&#x27;,
    moving_variance_initializer=&#x27;ones&#x27;, beta_regularizer=None,
    gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Batch normalization applies a transformation that maintains the mean output
close to 0 and the output standard deviation close to 1.

Importantly, batch normalization works differently during training and
during inference.

**During training** (i.e. when using `fit()` or when calling the layer/model
with the argument `training=True`), the layer normalizes its output using
the mean and standard deviation of the current batch of inputs. That is to
say, for each channel being normalized, the layer returns
`gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta`, where:

- `epsilon` is small constant (configurable as part of the constructor
arguments)
- `gamma` is a learned scaling factor (initialized as 1), which
can be disabled by passing `scale=False` to the constructor.
- `beta` is a learned offset factor (initialized as 0), which
can be disabled by passing `center=False` to the constructor.

**During inference** (i.e. when using `evaluate()` or `predict()` or when
calling the layer/model with the argument `training=False` (which is the
default), the layer normalizes its output using a moving average of the
mean and standard deviation of the batches it has seen during training. That
is to say, it returns
`gamma * (batch - self.moving_mean) / sqrt(self.moving_var + epsilon) + beta`.

`self.moving_mean` and `self.moving_var` are non-trainable variables that
are updated each time the layer in called in training mode, as such:

- `moving_mean = moving_mean * momentum + mean(batch) * (1 - momentum)`
- `moving_var = moving_var * momentum + var(batch) * (1 - momentum)`

As such, the layer will only normalize its inputs during inference
*after having been trained on data that has similar statistics as the
inference data*.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`axis`
</td>
<td>
Integer, the axis that should be normalized (typically the features
axis). For instance, after a `Conv2D` layer with
`data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
</td>
</tr><tr>
<td>
`momentum`
</td>
<td>
Momentum for the moving average.
</td>
</tr><tr>
<td>
`epsilon`
</td>
<td>
Small float added to variance to avoid dividing by zero.
</td>
</tr><tr>
<td>
`center`
</td>
<td>
If True, add offset of `beta` to normalized tensor. If False, `beta`
is ignored.
</td>
</tr><tr>
<td>
`scale`
</td>
<td>
If True, multiply by `gamma`. If False, `gamma` is not used. When the
next layer is linear (also e.g. <a href="../../../tf/nn/relu.md"><code>nn.relu</code></a>), this can be disabled since the
scaling will be done by the next layer.
</td>
</tr><tr>
<td>
`beta_initializer`
</td>
<td>
Initializer for the beta weight.
</td>
</tr><tr>
<td>
`gamma_initializer`
</td>
<td>
Initializer for the gamma weight.
</td>
</tr><tr>
<td>
`moving_mean_initializer`
</td>
<td>
Initializer for the moving mean.
</td>
</tr><tr>
<td>
`moving_variance_initializer`
</td>
<td>
Initializer for the moving variance.
</td>
</tr><tr>
<td>
`beta_regularizer`
</td>
<td>
Optional regularizer for the beta weight.
</td>
</tr><tr>
<td>
`gamma_regularizer`
</td>
<td>
Optional regularizer for the gamma weight.
</td>
</tr><tr>
<td>
`beta_constraint`
</td>
<td>
Optional constraint for the beta weight.
</td>
</tr><tr>
<td>
`gamma_constraint`
</td>
<td>
Optional constraint for the gamma weight.
</td>
</tr>
</table>



#### Call arguments:


* <b>`inputs`</b>: Input tensor (of any rank).
* <b>`training`</b>: Python boolean indicating whether the layer should behave in
  training mode or in inference mode.
  - `training=True`: The layer will normalize its inputs using the mean and
    variance of the current batch of inputs.
  - `training=False`: The layer will normalize its inputs using the mean and
    variance of its moving statistics, learned during training.


#### Input shape:

Arbitrary. Use the keyword argument `input_shape` (tuple of
integers, does not include the samples axis) when using this layer as the
first layer in a model.



#### Output shape:

Same shape as input.



#### Reference:

- [Ioffe and Szegedy, 2015](https://arxiv.org/abs/1502.03167).


**About setting `layer.trainable = False` on a `BatchNormalization` layer:**

The meaning of setting `layer.trainable = False` is to freeze the layer,
i.e. its internal state will not change during training:
its trainable weights will not be updated
during `fit()` or `train_on_batch()`, and its state updates will not be run.

Usually, this does not necessarily mean that the layer is run in inference
mode (which is normally controlled by the `training` argument that can
be passed when calling a layer). "Frozen state" and "inference mode"
are two separate concepts.

However, in the case of the `BatchNormalization` layer, **setting
`trainable = False` on the layer means that the layer will be
subsequently run in inference mode** (meaning that it will use
the moving mean and the moving variance to normalize the current batch,
rather than using the mean and variance of the current batch).

This behavior has been introduced in TensorFlow 2.0, in order
to enable `layer.trainable = False` to produce the most commonly
expected behavior in the convnet fine-tuning use case.

#### Note that:

- Setting `trainable` on an model containing other layers will
  recursively set the `trainable` value of all inner layers.
- If the value of the `trainable`
  attribute is changed after calling `compile()` on a model,
  the new value doesn't take effect for this model
  until `compile()` is called again.


