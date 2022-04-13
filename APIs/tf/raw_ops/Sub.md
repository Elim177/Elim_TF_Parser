description: Returns x - y element-wise.
robots: noindex

# tf.raw_ops.Sub

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Returns x - y element-wise.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.Sub`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.Sub(
    x, y, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

*NOTE*: <a href="../../tf/math/subtract.md"><code>tf.subtract</code></a> supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Both input and output have a range `(-inf, inf)`.

Example usages below.

Subtract operation between an array and a scalar:

```
>>> x = [1, 2, 3, 4, 5]
>>> y = 1
>>> tf.subtract(x, y)
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([0, 1, 2, 3, 4], dtype=int32)>
>>> tf.subtract(y, x)
<tf.Tensor: shape=(5,), dtype=int32,
numpy=array([ 0, -1, -2, -3, -4], dtype=int32)>
```

Note that binary `-` operator can be used instead:

```
>>> x = tf.convert_to_tensor([1, 2, 3, 4, 5])
>>> y = tf.convert_to_tensor(1)
>>> x - y
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([0, 1, 2, 3, 4], dtype=int32)>
```

Subtract operation between an array and a tensor of same shape:

```
>>> x = [1, 2, 3, 4, 5]
>>> y = tf.constant([5, 4, 3, 2, 1])
>>> tf.subtract(y, x)
<tf.Tensor: shape=(5,), dtype=int32,
numpy=array([ 4,  2,  0, -2, -4], dtype=int32)>
```

**Warning**: If one of the inputs (`x` or `y`) is a tensor and the other is a
non-tensor, the non-tensor input will adopt (or get casted to) the data type
of the tensor input. This can potentially cause unwanted overflow or underflow
conversion.

For example,

```
>>> x = tf.constant([1, 2], dtype=tf.int8)
>>> y = [2**8 + 1, 2**8 + 2]
>>> tf.subtract(x, y)
<tf.Tensor: shape=(2,), dtype=int8, numpy=array([0, 0], dtype=int8)>
```

When subtracting two input values of different shapes, <a href="../../tf/math/subtract.md"><code>tf.subtract</code></a> follows the
[general broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules)
. The two input array shapes are compared element-wise. Starting with the
trailing dimensions, the two dimensions either have to be equal or one of them
needs to be `1`.

For example,

```
>>> x = np.ones(6).reshape(2, 3, 1)
>>> y = np.ones(6).reshape(2, 1, 3)
>>> tf.subtract(x, y)
<tf.Tensor: shape=(2, 3, 3), dtype=float64, numpy=
array([[[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]],
       [[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]]])>
```

Example with inputs of different dimensions:

```
>>> x = np.ones(6).reshape(2, 3, 1)
>>> y = np.ones(6).reshape(1, 6)
>>> tf.subtract(x, y)
<tf.Tensor: shape=(2, 3, 6), dtype=float64, numpy=
array([[[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]],
       [[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]]])>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`
</td>
<td>
A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `complex64`, `complex128`, `uint32`, `uint64`.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
A `Tensor`. Must have the same type as `x`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor`. Has the same type as `x`.
</td>
</tr>

</table>

