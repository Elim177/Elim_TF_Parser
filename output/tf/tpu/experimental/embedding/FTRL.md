description: Optimization parameters for FTRL with TPU embeddings.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.tpu.experimental.embedding.FTRL" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.tpu.experimental.embedding.FTRL

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/tpu/tpu_embedding_v2_utils.py">View source</a>



Optimization parameters for FTRL with TPU embeddings.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.tpu.experimental.embedding.FTRL`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.tpu.experimental.embedding.FTRL(
    learning_rate: Union[float, Callable[[], float]] = 0.001,
    learning_rate_power: float = -0.5,
    l1_regularization_strength: float = 0.0,
    l2_regularization_strength: float = 0.0,
    beta: float = 0.0,
    initial_accumulator_value: float = 0.1,
    use_gradient_accumulation: bool = (True),
    clip_weight_min: Optional[float] = None,
    clip_weight_max: Optional[float] = None,
    weight_decay_factor: Optional[float] = None,
    multiply_weight_decay_factor_by_learning_rate: bool = None,
    slot_variable_creation_fn: Optional[SlotVarCreationFnType] = None,
    clipvalue: Optional[ClipValueType] = None,
    multiply_linear_by_learning_rate: bool = (False),
    allow_zero_accumulator: bool = (False)
)
</code></pre>



<!-- Placeholder for "Used in" -->

See Algorithm 1 of this
[paper](https://research.google.com/pubs/archive/41159.pdf).

Pass this to <a href="../../../../tf/tpu/experimental/embedding/TPUEmbedding.md"><code>tf.tpu.experimental.embedding.TPUEmbedding</code></a> via the `optimizer`
argument to set the global optimizer and its parameters:

```python
embedding = tf.tpu.experimental.embedding.TPUEmbedding(
    ...
    optimizer=tf.tpu.experimental.embedding.FTRL(0.1))
```

This can also be used in a <a href="../../../../tf/tpu/experimental/embedding/TableConfig.md"><code>tf.tpu.experimental.embedding.TableConfig</code></a> as the
optimizer parameter to set a table specific optimizer. This will override the
optimizer and parameters for global embedding optimizer defined above:

```python
table_one = tf.tpu.experimental.embedding.TableConfig(
    vocabulary_size=...,
    dim=...,
    optimizer=tf.tpu.experimental.embedding.FTRL(0.2))
table_two = tf.tpu.experimental.embedding.TableConfig(
    vocabulary_size=...,
    dim=...)

feature_config = (
    tf.tpu.experimental.embedding.FeatureConfig(
        table=table_one),
    tf.tpu.experimental.embedding.FeatureConfig(
        table=table_two))

embedding = tf.tpu.experimental.embedding.TPUEmbedding(
    feature_config=feature_config,
    batch_size=...
    optimizer=tf.tpu.experimental.embedding.FTRL(0.1))
```

In the above example, the first feature will be looked up in a table that has
a learning rate of 0.2 while the second feature will be looked up in a table
that has a learning rate of 0.1.

See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
complete description of these parameters and their impacts on the optimizer
algorithm.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`learning_rate`
</td>
<td>
The learning rate. It should be a floating point value or a
callable taking no arguments for a dynamic learning rate.
</td>
</tr><tr>
<td>
`learning_rate_power`
</td>
<td>
A float value, must be less or equal to zero.
Controls how the learning rate decreases during training. Use zero for a
fixed learning rate.
</td>
</tr><tr>
<td>
`l1_regularization_strength`
</td>
<td>
A float value, must be greater than or equal
to zero.
</td>
</tr><tr>
<td>
`l2_regularization_strength`
</td>
<td>
A float value, must be greater than or equal
to zero.
</td>
</tr><tr>
<td>
`beta`
</td>
<td>
A float value, representing the beta value from the paper.
</td>
</tr><tr>
<td>
`initial_accumulator_value`
</td>
<td>
The starting value for accumulators. Only zero
or positive values are allowed.
</td>
</tr><tr>
<td>
`use_gradient_accumulation`
</td>
<td>
setting this to `False` makes embedding
gradients calculation less accurate but faster.
</td>
</tr><tr>
<td>
`clip_weight_min`
</td>
<td>
the minimum value to clip by; None means -infinity.
</td>
</tr><tr>
<td>
`clip_weight_max`
</td>
<td>
the maximum value to clip by; None means +infinity.
</td>
</tr><tr>
<td>
`weight_decay_factor`
</td>
<td>
amount of weight decay to apply; None means that the
weights are not decayed.
</td>
</tr><tr>
<td>
`multiply_weight_decay_factor_by_learning_rate`
</td>
<td>
if true,
`weight_decay_factor` is multiplied by the current learning rate.
</td>
</tr><tr>
<td>
`slot_variable_creation_fn`
</td>
<td>
If you wish do directly control the creation of
the slot variables, set this to a callable taking three parameters: a
  table variable, a list of slot names to create for it, and a list of
  initializers. This function should return a dict with the slot names
  as keys and the created variables as values with types matching the
  table variable. When set to None (the default), uses the built-in
  variable creation.
</td>
</tr><tr>
<td>
`clipvalue`
</td>
<td>
Controls clipping of the gradient. Set to either a single
positive scalar value to get clipping or a tuple of scalar values (min,
max) to set a separate maximum or minimum. If one of the two entries is
None, then there will be no clipping that direction.
</td>
</tr><tr>
<td>
`multiply_linear_by_learning_rate`
</td>
<td>
If set to True, a modified formula is
used for FTRL that treats the "linear" accumulator as being
pre-multiplied by the learning rate (i.e., the accumulator named
"linear" actually stores "linear * learning_rate"). Other than
checkpoint compatibility, this is mathematically equivalent for a static
learning rate; for a dynamic learning rate, it is nearly the same as
long as the learning rate does not change quickly. The benefit of this
is that the modified formula handles zero and near-zero learning rates
without producing NaNs, improving flexibility for learning rate ramp-up.
</td>
</tr><tr>
<td>
`allow_zero_accumulator`
</td>
<td>
If set to True, changes some internal formulas to
allow zero and near-zero accumulator values at the cost of some
performance; this only needs to be set if you are using an initial
accumulator value of zero, which is uncommon.
</td>
</tr>
</table>



