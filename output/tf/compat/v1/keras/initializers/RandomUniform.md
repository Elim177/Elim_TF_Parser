description: Initializer that generates tensors with a uniform distribution.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.keras.initializers.RandomUniform" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tf.compat.v1.keras.initializers.RandomUniform

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/initializers/initializers_v1.py#L163-L278">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Initializer that generates tensors with a uniform distribution.

Inherits From: [`random_uniform_initializer`](../../../../../tf/compat/v1/random_uniform_initializer.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.initializers.random_uniform`, `tf.compat.v1.keras.initializers.uniform`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.keras.initializers.RandomUniform(
    minval=-0.05, maxval=0.05, seed=None, dtype=tf.dtypes.float32
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

Although it is a legacy <a href="../../../../../tf/compat/v1.md"><code>compat.v1</code></a> api,
<a href="../../../../../tf/compat/v1/keras/initializers/RandomUniform.md"><code>tf.compat.v1.keras.initializers.RandomUniform</code></a> is compatible with eager
execution and <a href="../../../../../tf/function.md"><code>tf.function</code></a>.

To switch to native TF2, switch to using
<a href="../../../../../tf/keras/initializers/RandomUniform.md"><code>tf.keras.initializers.RandomUniform</code></a> (not from <a href="../../../../../tf/compat/v1.md"><code>compat.v1</code></a>) and
if you need to change the default dtype use
<a href="../../../../../tf/keras/backend/set_floatx.md"><code>tf.keras.backend.set_floatx(float_dtype)</code></a>
or pass the dtype when calling the initializer, rather than passing it
when constructing the initializer.

Random seed behavior:

Also be aware that if you pass a seed to the TF2 initializer
API it will reuse that same seed for every single initialization
(unlike the TF1 intializer)

#### Structural Mapping to Native TF2

Before:

```python

initializer = tf.compat.v1.keras.initializers.RandomUniform(
  minval=minval,
  maxval=maxval,
  seed=seed,
  dtype=dtype)

weight_one = tf.Variable(initializer(shape_one))
weight_two = tf.Variable(initializer(shape_two))
```

After:

```python
initializer = tf.keras.initializers.RandomUniform(
  minval=minval,
  maxval=maxval,
  # seed=seed,  # Setting a seed in the native TF2 API
                # causes it to produce the same initializations
                # across multiple calls of the same initializer.
  )

weight_one = tf.Variable(initializer(shape_one, dtype=dtype))
weight_two = tf.Variable(initializer(shape_two, dtype=dtype))
```

#### How to Map Arguments

| TF1 Arg Name      | TF2 Arg Name    | Note                       |
| :---------------- | :-------------- | :------------------------- |
| `minval`            | `minval`          | No change to defaults |
| `maxval`          | `maxval`        | No change to defaults |
| `seed`            | `seed`          | Different random number generation |
:                    :        : semantics (to change in a :
:                    :        : future version). If set, the TF2 version :
:                    :        : will use stateless random number :
:                    :        : generation which will produce the exact :
:                    :        : same initialization even across multiple :
:                    :        : calls of the initializer instance. the :
:                    :        : <a href="../../../../../tf/compat/v1.md"><code>compat.v1</code></a> version will generate new :
:                    :        : initializations each time. Do not set :
:                    :        : a seed if you need different          :
:                    :        : initializations each time. Instead    :
:                    :        : either set a global tf seed with
:                    :        : <a href="../../../../../tf/random/set_seed.md"><code>tf.random.set_seed</code></a> if you need :
:                    :        : determinism, or initialize each weight :
:                    :        : with a separate initializer instance  :
:                    :        : and a different seed.                 :
| `dtype`           | `dtype`  | The TF2 native api only takes it  |
:                   :      : as a `__call__` arg, not a constructor arg. :
| `partition_info`  | -    |  (`__call__` arg in TF1) Not supported      |

#### Example of fixed-seed behavior differences

<a href="../../../../../tf/compat/v1.md"><code>compat.v1</code></a> Fixed seed behavior:

```
>>> initializer = tf.compat.v1.keras.initializers.RandomUniform(seed=10)
>>> a = initializer(shape=(2, 2))
>>> b = initializer(shape=(2, 2))
>>> tf.reduce_sum(a - b) == 0
<tf.Tensor: shape=(), dtype=bool, numpy=False>
```

After:

```
>>> initializer = tf.keras.initializers.RandomUniform(seed=10)
>>> a = initializer(shape=(2, 2))
>>> b = initializer(shape=(2, 2))
>>> tf.reduce_sum(a - b) == 0
<tf.Tensor: shape=(), dtype=bool, numpy=True>
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
`minval`
</td>
<td>
A python scalar or a scalar tensor. Lower bound of the range of
random values to generate.
</td>
</tr><tr>
<td>
`maxval`
</td>
<td>
A python scalar or a scalar tensor. Upper bound of the range of
random values to generate.  Defaults to 1 for float types.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
A Python integer. Used to create random seeds. See
<a href="../../../../../tf/compat/v1/set_random_seed.md"><code>tf.compat.v1.set_random_seed</code></a> for behavior.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
Default data type, used if no `dtype` argument is provided when
calling the initializer.
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





