description: Asserts that the given Tensor is of the specified type.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.debugging.assert_type" />
<meta itemprop="path" content="Stable" />
</div>

# tf.debugging.assert_type

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/check_ops.py">View source</a>



Asserts that the given `Tensor` is of the specified type.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.debugging.assert_type(
    tensor, tf_type, message=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This can always be checked statically, so this method returns nothing.

#### Example:



```
>>> a = tf.Variable(1.0)
>>> tf.debugging.assert_type(a, tf_type= tf.float32)
```

```
>>> b = tf.constant(21)
>>> tf.debugging.assert_type(b, tf_type=tf.bool)
Traceback (most recent call last):
...
TypeError: ...
```

```
>>> c = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2],
...  dense_shape=[3, 4])
>>> tf.debugging.assert_type(c, tf_type= tf.int32)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensor`
</td>
<td>
A `Tensor`, `SparseTensor` or `tf.Variable .
</td>
</tr><tr>
<td>
`tf_type`
</td>
<td>
A tensorflow type (<a href="../../tf/dtypes.md#float32"><code>dtypes.float32</code></a>, <a href="../../tf.md#int64"><code>tf.int64</code></a>, <a href="../../tf/dtypes.md#bool"><code>dtypes.bool</code></a>,
etc).
</td>
</tr><tr>
<td>
`message`
</td>
<td>
A string to prefix to the default message.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
 A name for this operation. Defaults to "assert_type"
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
If the tensor's data type doesn't match `tf_type`.
</td>
</tr>
</table>

