description: Optimizer that implements the FTRL algorithm.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.optimizers.Ftrl" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.optimizers.Ftrl

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/optimizer_v2/ftrl.py#L24-L265">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Optimizer that implements the FTRL algorithm.

Inherits From: [`Optimizer`](../../../tf/keras/optimizers/Optimizer.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.optimizers.Ftrl`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.optimizers.Ftrl`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.optimizers.Ftrl(
    learning_rate=0.001, learning_rate_power=-0.5, initial_accumulator_value=0.1,
    l1_regularization_strength=0.0, l2_regularization_strength=0.0,
    name=&#x27;Ftrl&#x27;, l2_shrinkage_regularization_strength=0.0, beta=0.0,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

"Follow The Regularized Leader" (FTRL) is an optimization algorithm developed
at Google for click-through rate prediction in the early 2010s. It is most
suitable for shallow models with large and sparse feature spaces.
The algorithm is described by
[McMahan et al., 2013](https://research.google.com/pubs/archive/41159.pdf).
The Keras version has support for both online L2 regularization
(the L2 regularization described in the paper
above) and shrinkage-type L2 regularization
(which is the addition of an L2 penalty to the loss function).

#### Initialization:



```python
n = 0
sigma = 0
z = 0
```

Update rule for one variable `w`:

```python
prev_n = n
n = n + g ** 2
sigma = (sqrt(n) - sqrt(prev_n)) / lr
z = z + g - sigma * w
if abs(z) < lambda_1:
  w = 0
else:
  w = (sgn(z) * lambda_1 - z) / ((beta + sqrt(n)) / alpha + lambda_2)
```

#### Notation:



- `lr` is the learning rate
- `g` is the gradient for the variable
- `lambda_1` is the L1 regularization strength
- `lambda_2` is the L2 regularization strength

Check the documentation for the `l2_shrinkage_regularization_strength`
parameter for more details when shrinkage is enabled, in which case gradient
is replaced with a gradient with shrinkage.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`learning_rate`
</td>
<td>
A `Tensor`, floating point value, or a schedule that is a
<a href="../../../tf/keras/optimizers/schedules/LearningRateSchedule.md"><code>tf.keras.optimizers.schedules.LearningRateSchedule</code></a>. The learning rate.
</td>
</tr><tr>
<td>
`learning_rate_power`
</td>
<td>
A float value, must be less or equal to zero.
Controls how the learning rate decreases during training. Use zero for
a fixed learning rate.
</td>
</tr><tr>
<td>
`initial_accumulator_value`
</td>
<td>
The starting value for accumulators.
Only zero or positive values are allowed.
</td>
</tr><tr>
<td>
`l1_regularization_strength`
</td>
<td>
A float value, must be greater than or
equal to zero. Defaults to 0.0.
</td>
</tr><tr>
<td>
`l2_regularization_strength`
</td>
<td>
A float value, must be greater than or
equal to zero. Defaults to 0.0.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optional name prefix for the operations created when applying
gradients.  Defaults to `"Ftrl"`.
</td>
</tr><tr>
<td>
`l2_shrinkage_regularization_strength`
</td>
<td>
A float value, must be greater than
or equal to zero. This differs from L2 above in that the L2 above is a
stabilization penalty, whereas this L2 shrinkage is a magnitude penalty.
When input is sparse shrinkage will only happen on the active weights.
</td>
</tr><tr>
<td>
`beta`
</td>
<td>
A float value, representing the beta value from the paper.
Defaults to 0.0.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword arguments. Allowed to be one of
`"clipnorm"` or `"clipvalue"`.
`"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
gradients by value.
</td>
</tr>
</table>



#### Reference:

- [McMahan et al., 2013](
  https://research.google.com/pubs/archive/41159.pdf)


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
in case of any invalid argument.
</td>
</tr>
</table>



