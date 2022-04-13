description: Returns x + y element-wise.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.add" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.add

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>



Returns x + y element-wise.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.add`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.add`, `tf.compat.v1.math.add`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.add(
    x, y, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Example usages below.

Add a scalar and a list:

```
>>> x = [1, 2, 3, 4, 5]
>>> y = 1
>>> tf.add(x, y)
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([2, 3, 4, 5, 6],
dtype=int32)>
```

Note that binary `+` operator can be used instead:

```
>>> x = tf.convert_to_tensor([1, 2, 3, 4, 5])
>>> y = tf.convert_to_tensor(1)
>>> x + y
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([2, 3, 4, 5, 6],
dtype=int32)>
```

Add a tensor and a list of same shape:

```
>>> x = [1, 2, 3, 4, 5]
>>> y = tf.constant([1, 2, 3, 4, 5])
>>> tf.add(x, y)
<tf.Tensor: shape=(5,), dtype=int32,
numpy=array([ 2,  4,  6,  8, 10], dtype=int32)>
```

**Warning**: If one of the inputs (`x` or `y`) is a tensor and the other is a
non-tensor, the non-tensor input will adopt (or get casted to) the data type
of the tensor input. This can potentially cause unwanted overflow or underflow
conversion.

For example,

```
>>> x = tf.constant([1, 2], dtype=tf.int8)
>>> y = [2**7 + 1, 2**7 + 2]
>>> tf.add(x, y)
<tf.Tensor: shape=(2,), dtype=int8, numpy=array([-126, -124], dtype=int8)>
```

When adding two input values of different shapes, `Add` follows NumPy
broadcasting rules. The two input array shapes are compared element-wise.
Starting with the trailing dimensions, the two dimensions either have to be
equal or one of them needs to be `1`.

For example,

```
>>> x = np.ones(6).reshape(1, 2, 1, 3)
>>> y = np.ones(6).reshape(2, 1, 3, 1)
>>> tf.add(x, y).shape.as_list()
[2, 2, 3, 3]
```

Another example with two arrays of different dimension.

```
>>> x = np.ones([1, 2, 1, 4])
>>> y = np.ones([3, 4])
>>> tf.add(x, y).shape.as_list()
[1, 2, 3, 4]
```

The reduction version of this elementwise operation is <a href="../../tf/math/reduce_sum.md"><code>tf.math.reduce_sum</code></a>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`
</td>
<td>
A <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a>. Must be one of the following types: bfloat16, half,
float32, float64, uint8, int8, int16, int32, int64, complex64, complex128,
string.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
A <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a>. Must have the same type as x.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional)
</td>
</tr>
</table>

