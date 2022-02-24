description: Initializer that generates tensors with a normal distribution.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.random_normal_initializer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tf.compat.v1.random_normal_initializer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/init_ops.py">View source</a>



Initializer that generates tensors with a normal distribution.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.initializers.random_normal`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.random_normal_initializer(
    mean=0.0, stddev=1.0, seed=None, dtype=tf.dtypes.float32
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

Although it is a legacy <a href="../../../tf/compat/v1.md"><code>compat.v1</code></a> API, this symbol is compatible with eager
execution and <a href="../../../tf/function.md"><code>tf.function</code></a>.

To switch to native TF2, switch to using either
<a href="../../../tf/keras/initializers/RandomNormal.md"><code>tf.initializers.RandomNormal</code></a> or <a href="../../../tf/keras/initializers/RandomNormal.md"><code>tf.keras.initializers.RandomNormal</code></a>
(neither from <a href="../../../tf/compat/v1.md"><code>compat.v1</code></a>) and
pass the dtype when calling the initializer. Keep in mind that
the default stddev and the behavior of fixed seeds have changed.

#### Structural Mapping to Native TF2

Before:

```python
initializer = tf.compat.v1.random_normal_initializer(
  mean=mean,
  stddev=stddev,
  seed=seed,
  dtype=dtype)

weight_one = tf.Variable(initializer(shape_one))
weight_two = tf.Variable(initializer(shape_two))
```

After:

```python
initializer = tf.initializers.RandomNormal(
  mean=mean,
  seed=seed,
  stddev=stddev)

weight_one = tf.Variable(initializer(shape_one, dtype=dtype))
weight_two = tf.Variable(initializer(shape_two, dtype=dtype))
```

#### How to Map Arguments

| TF1 Arg Name       | TF2 Arg Name    | Note                       |
| :----------------- | :-------------- | :------------------------- |
| `mean`             | `mean`          | No change to defaults |
| `stddev`           | `stddev`        | Default changes from 1.0 to 0.05 |
| `seed`             | `seed`          |                                  |
| `dtype`            | `dtype`  | The TF2 native api only takes it as a |
:                    :          : `__call__` arg, not a constructor arg. :
| `partition_info`   | -     |  (`__call__` arg in TF1) Not supported.  |



 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mean`
</td>
<td>
a python scalar or a scalar tensor. Mean of the random values to
generate.
</td>
</tr><tr>
<td>
`stddev`
</td>
<td>
a python scalar or a scalar tensor. Standard deviation of the random
values to generate.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
A Python integer. Used to create random seeds. See
<a href="../../../tf/compat/v1/set_random_seed.md"><code>tf.compat.v1.set_random_seed</code></a> for behavior.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
Default data type, used if no `dtype` argument is provided when
calling the initializer. Only floating point types are supported.
</td>
</tr>
</table>




## Methods

<h3 id="from_config"><code>from_config</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/ops/init_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_config(
    config
)
</code></pre>

Instantiates an initializer from a configuration dictionary.


#### Example:



```python
initializer = RandomUniform(-1, 1)
config = initializer.get_config()
initializer = RandomUniform.from_config(config)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`config`
</td>
<td>
A Python dictionary. It will typically be the output of
`get_config`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An Initializer instance.
</td>
</tr>

</table>



<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/ops/init_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

Returns the configuration of the initializer as a JSON-serializable dict.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A JSON-serializable Python dict.
</td>
</tr>

</table>



<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/ops/init_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    shape, dtype=None, partition_info=None
)
</code></pre>

Returns a tensor object initialized as specified by the initializer.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`shape`
</td>
<td>
Shape of the tensor.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
Optional dtype of the tensor. If not provided use the initializer
dtype.
</td>
</tr><tr>
<td>
`partition_info`
</td>
<td>
Optional information about the possible partitioning of a
tensor.
</td>
</tr>
</table>





