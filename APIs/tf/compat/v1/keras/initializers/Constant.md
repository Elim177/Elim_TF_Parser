description: Initializer that generates tensors with constant values.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.keras.initializers.Constant" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tf.compat.v1.keras.initializers.Constant

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/init_ops.py">View source</a>



Initializer that generates tensors with constant values.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.constant_initializer`, `tf.compat.v1.initializers.constant`, `tf.compat.v1.keras.initializers.constant`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.keras.initializers.Constant(
    value=0, dtype=tf.dtypes.float32, verify_shape=(False)
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

Although it is a legacy API endpoint, <a href="../../../../../tf/compat/v1/keras/initializers/Constant.md"><code>tf.compat.v1.constant_initializer</code></a>
is compatible with eager execution and <a href="../../../../../tf/function.md"><code>tf.function</code></a>.

To migrate to a non-legacy TF2 API, please use <a href="../../../../../tf/constant_initializer.md"><code>tf.constant_initializer</code></a>
instead. The `dtype`
argument in <a href="../../../../../tf/compat/v1/keras/initializers/Constant.md#__init__"><code>tf.compat.v1.constant_initializer.__init__()</code></a> does not exist in
<a href="../../../../../tf/constant_initializer.md#__init__"><code>tf.constant_initializer.__init__()</code></a>. However, you can specify the `dtype` in
`__call__()` in both cases.

In the <a href="../../../../../tf/compat/v1.md"><code>compat.v1</code></a> symbol, if `verify_shape` is set to `True`, an exception
is raised when initializing a variable with a different shape from
`value`. If set to `False`, `value` is reshaped to initialize the variable
if necessary. An exception would only be raised when the number of
elements are different.

The `verify_shape` argument is not supported in TF2. Using
<a href="../../../../../tf/constant_initializer.md"><code>tf.constant_initializer</code></a> is equivalent to setting `verify_shape` to `False`.

#### Structural Mapping to Native TF2

Before:

```python
value = [0, 1, 2, 3, 4, 5, 6, 7]
initializer = tf.compat.v1.constant_initializer(
    value=value,
    dtype=tf.float32,
    verify_shape=False)
variable = tf.Variable(initializer(shape=[2, 4]))
```

After:

```python
value = [0, 1, 2, 3, 4, 5, 6, 7]
initializer = tf.constant_initializer(value=value)
tf.Variable(initializer(shape=[2, 4], dtype=tf.float32))
```

#### How to Map Arguments

| TF1 Arg Name          | TF2 Arg Name     | Note                        |
| :-------------------- | :--------------- | :-------------------------- |
| `value`               | `value`          | In constructor              |
| `dtype`               | `dtype`          | In `__call__()` method      |
| `verify_shape`        | Not Supported    | Equivalent to set to `False`|
| `partition_info`      | - |  (`__call__` arg in TF1) Not supported     |


#### Before & After Usage Example

Before:

```
>>> value = [1., 2., 3., 4.]
>>> initializer = tf.compat.v1.constant_initializer(
...     value=value, dtype=tf.float32, verify_shape=True)
>>> tf.Variable(initializer(shape=[2, 2])).numpy()
Traceback (most recent call last):
...
TypeError: Expected Tensor's shape: (2, 2), got (4,).
>>> initializer = tf.compat.v1.constant_initializer(
...     value=value, dtype=tf.float32, verify_shape=False)
>>> tf.Variable(initializer(shape=[2, 2])).numpy()
array([[1., 2.],
       [3., 4.]], dtype=float32)
```

After:

```
>>> value = [1., 2., 3., 4.]
>>> initializer = tf.constant_initializer(value=value)
>>> tf.Variable(initializer(shape=[2, 2], dtype=tf.float32)).numpy()
array([[1., 2.],
       [3., 4.]], dtype=float32)
```



 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

The resulting tensor is populated with values of type `dtype`, as
specified by arguments `value` following the desired `shape` of the
new tensor (see examples below).

The argument `value` can be a constant value, or a list of values of type
`dtype`. If `value` is a list, then the length of the list must be less
than or equal to the number of elements implied by the desired shape of the
tensor. In the case where the total number of elements in `value` is less
than the number of elements required by the tensor shape, the last element
in `value` will be used to fill the remaining entries. If the total number of
elements in `value` is greater than the number of elements required by the
tensor shape, the initializer will raise a `ValueError`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`value`
</td>
<td>
A Python scalar, list or tuple of values, or a N-dimensional numpy
array. All elements of the initialized variable will be set to the
corresponding value in the `value` argument.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
Default data type, used if no `dtype` argument is provided when
calling the initializer.
</td>
</tr><tr>
<td>
`verify_shape`
</td>
<td>
Boolean that enables verification of the shape of `value`. If
`True`, the initializer will throw an error if the shape of `value` is not
compatible with the shape of the initialized tensor.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
If the input `value` is not one of the expected types.
</td>
</tr>
</table>



#### Examples:

The following example can be rewritten using a numpy.ndarray instead
of the `value` list, even reshaped, as shown in the two commented lines
below the `value` list initialization.


```
>>> value = [0, 1, 2, 3, 4, 5, 6, 7]
>>> init = tf.compat.v1.constant_initializer(value)
>>> # fitting shape
>>> with tf.compat.v1.Session():
...   x = tf.compat.v1.get_variable('x', shape=[2, 4], initializer=init)
...   x.initializer.run()
...   print(x.eval())
[[0. 1. 2. 3.]
 [4. 5. 6. 7.]]
>>> # Larger shape
>>> with tf.compat.v1.Session():
...   y = tf.compat.v1.get_variable('y', shape=[3, 4], initializer=init)
...   y.initializer.run()
...   print(y.eval())
[[0.  1.  2.  3.]
 [4.  5.  6.  7.]
 [7.  7.  7.  7.]]
>>> # Smaller shape
>>> with tf.compat.v1.Session():
...   z = tf.compat.v1.get_variable('z', shape=[2, 3], initializer=init)
Traceback (most recent call last):
...
ValueError: Too many elements provided. Needed at most 6, but received 8
>>> # Shape verification
>>> init_verify = tf.compat.v1.constant_initializer(value, verify_shape=True)
>>> with tf.compat.v1.Session():
...  u = tf.compat.v1.get_variable('u', shape=[3, 4],
...                                initializer=init_verify)
Traceback (most recent call last):
...
TypeError: Expected Tensor's shape: (3, 4), got (8,).
```



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
    shape, dtype=None, partition_info=None, verify_shape=None
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





