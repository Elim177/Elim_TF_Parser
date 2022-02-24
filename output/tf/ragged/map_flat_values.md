description: Applies op to the flat_values of one or more RaggedTensors.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.ragged.map_flat_values" />
<meta itemprop="path" content="Stable" />
</div>

# tf.ragged.map_flat_values

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/ragged/ragged_functional_ops.py">View source</a>



Applies `op` to the `flat_values` of one or more RaggedTensors.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.ragged.map_flat_values`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.ragged.map_flat_values(
    op, *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Replaces any `RaggedTensor` in `args` or `kwargs` with its `flat_values`
tensor (which collapses all ragged dimensions), and then calls `op`.  Returns
a `RaggedTensor` that is constructed from the input `RaggedTensor`s'
`nested_row_splits` and the value returned by the `op`.

If the input arguments contain multiple `RaggedTensor`s, then they must have
identical `nested_row_splits`.

This operation is generally used to apply elementwise operations to each value
in a `RaggedTensor`.

Warning: <a href="../../tf/ragged/map_flat_values.md"><code>tf.ragged.map_flat_values</code></a> does *not* apply `op` to each row of a
ragged tensor.  This difference is important for non-elementwise operations,
such as <a href="../../tf/math/reduce_sum.md"><code>tf.reduce_sum</code></a>.  If you wish to apply a non-elementwise operation to
each row of a ragged tensor, use <a href="../../tf/map_fn.md"><code>tf.map_fn</code></a> instead.  (You may need to
specify an `output_signature` when using <a href="../../tf/map_fn.md"><code>tf.map_fn</code></a> with ragged tensors.)

#### Examples:



```
>>> rt = tf.ragged.constant([[1, 2, 3], [], [4, 5], [6]])
>>> tf.ragged.map_flat_values(tf.ones_like, rt)
<tf.RaggedTensor [[1, 1, 1], [], [1, 1], [1]]>
>>> tf.ragged.map_flat_values(tf.multiply, rt, rt)
<tf.RaggedTensor [[1, 4, 9], [], [16, 25], [36]]>
>>> tf.ragged.map_flat_values(tf.add, rt, 5)
<tf.RaggedTensor [[6, 7, 8], [], [9, 10], [11]]>
```

Example with a non-elementwise operation (note that `map_flat_values` and
`map_fn` return different results):

```
>>> rt = tf.ragged.constant([[1.0, 3.0], [], [3.0, 6.0, 3.0]])
>>> def normalized(x):
...   return x / tf.reduce_sum(x)
>>> tf.ragged.map_flat_values(normalized, rt)
<tf.RaggedTensor [[0.0625, 0.1875], [], [0.1875, 0.375, 0.1875]]>
>>> tf.map_fn(normalized, rt)
<tf.RaggedTensor [[0.25, 0.75], [], [0.25, 0.5, 0.25]]>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`op`
</td>
<td>
The operation that should be applied to the RaggedTensor `flat_values`.
`op` is typically an element-wise operation (such as math_ops.add), but
any operation that preserves the size of the outermost dimension can be
used.  I.e., `shape[0]` of the value returned by `op` must match
`shape[0]` of the `RaggedTensor`s' `flat_values` tensors.
</td>
</tr><tr>
<td>
`*args`
</td>
<td>
Arguments for `op`.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword arguments for `op`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `RaggedTensor` whose `ragged_rank` matches the `ragged_rank` of all
input `RaggedTensor`s.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If args contains no `RaggedTensors`, or if the `nested_splits`
of the input `RaggedTensor`s are not identical.
</td>
</tr>
</table>

