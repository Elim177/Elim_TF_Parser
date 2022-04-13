description: Initializer that generates tensors initialized to 0.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.keras.initializers.Zeros" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tf.compat.v1.keras.initializers.Zeros

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/init_ops.py">View source</a>



Initializer that generates tensors initialized to 0.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.initializers.zeros`, `tf.compat.v1.keras.initializers.zeros`, `tf.compat.v1.zeros_initializer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.keras.initializers.Zeros(
    dtype=tf.dtypes.float32
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

<a href="../../../../../tf/compat/v1/keras/initializers/Zeros.md"><code>tf.compat.v1.zeros_initializer</code></a> is compatible with eager execution
and <a href="../../../../../tf/function.md"><code>tf.function</code></a>.

To migrate to TF2, please use <a href="../../../../../tf/zeros_initializer.md"><code>tf.zeros_initializer</code></a> instead. The `dtype`
argument in <a href="../../../../../tf/compat/v1/keras/initializers/Zeros.md#__init__"><code>tf.compat.v1.zeros_initializer.__init__()</code></a> does not exist in
<a href="../../../../../tf/zeros_initializer.md#__init__"><code>tf.zeros_initializer.__init__()</code></a>. However, you can specify the `dtype` in
`__call__()` in both cases.

#### Structural Mapping to Native TF2

Before:

```python
initializer = tf.compat.v1.zeros_initializer(dtype=tf.float32)
variable = tf.Variable(initializer(shape=[3, 3]))
```

After:

```python
initializer = tf.zeros_initializer()
variable = tf.Variable(initializer(shape=[3, 3], dtype=tf.float32))
```

#### How to Map Arguments

| TF1 Arg Name         | TF2 Arg Name     | Note                       |
| :------------------- | :--------------- | :------------------------- |
| `dtype`              | `dtype`          | In `__call__()` method     |
| `partition_info`     | - |  (`__call__` arg in TF1) Not supported    |


#### Before & After Usage Example

Before:

```
>>> initializer = tf.compat.v1.zeros_initializer(dtype=tf.float32)
>>> tf.Variable(initializer(shape=[3])).numpy()
array([0., 0., 0.], dtype=float32)
>>> tf.Variable(initializer(shape=[3, 3])).numpy()
array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]], dtype=float32)
>>> initializer = tf.compat.v1.zeros_initializer()
>>> tf.Variable(initializer(shape=[3], dtype=tf.float32)).numpy()
array([0., 0., 0.], dtype=float32)
>>> tf.Variable(initializer(shape=[3, 3], dtype=tf.float32)).numpy()
array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]], dtype=float32)
```

After:

```
>>> initializer = tf.zeros_initializer()
>>> tf.Variable(initializer(shape=[3], dtype=tf.float32)).numpy()
array([0., 0., 0.], dtype=float32)
>>> tf.Variable(initializer(shape=[3, 3], dtype=tf.float32)).numpy()
array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]], dtype=float32)
```



 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->



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





