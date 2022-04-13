description: Returns the indices of a tensor that give its sorted order along an axis.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.argsort" />
<meta itemprop="path" content="Stable" />
</div>

# tf.argsort

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/sort_ops.py">View source</a>



Returns the indices of a tensor that give its sorted order along an axis.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.argsort`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.argsort(
    values, axis=-1, direction=&#x27;ASCENDING&#x27;, stable=(False), name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

```
>>> values = [1, 10, 26.9, 2.8, 166.32, 62.3]
>>> sort_order = tf.argsort(values)
>>> sort_order.numpy()
array([0, 3, 1, 2, 5, 4], dtype=int32)
```

#### For a 1D tensor:



```
>>> sorted = tf.gather(values, sort_order)
>>> assert tf.reduce_all(sorted == tf.sort(values))
```

For higher dimensions, the output has the same shape as
`values`, but along the given axis, values represent the index of the sorted
element in that slice of the tensor at the given position.

```
>>> mat = [[30,20,10],
...        [20,10,30],
...        [10,30,20]]
>>> indices = tf.argsort(mat)
>>> indices.numpy()
array([[2, 1, 0],
       [1, 0, 2],
       [0, 2, 1]], dtype=int32)
```

If `axis=-1` these indices can be used to apply a sort using `tf.gather`:

```
>>> tf.gather(mat, indices, batch_dims=-1).numpy()
array([[10, 20, 30],
       [10, 20, 30],
       [10, 20, 30]], dtype=int32)
```

#### See also:


* <a href="../tf/sort.md"><code>tf.sort</code></a>: Sort along an axis.
* <a href="../tf/math/top_k.md"><code>tf.math.top_k</code></a>: A partial sort that returns a fixed number of top values
  and corresponding indices.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`values`
</td>
<td>
1-D or higher **numeric** `Tensor`.
</td>
</tr><tr>
<td>
`axis`
</td>
<td>
The axis along which to sort. The default is -1, which sorts the last
axis.
</td>
</tr><tr>
<td>
`direction`
</td>
<td>
The direction in which to sort the values (`'ASCENDING'` or
`'DESCENDING'`).
</td>
</tr><tr>
<td>
`stable`
</td>
<td>
If True, equal elements in the original tensor will not be
re-ordered in the returned order. Unstable sort is not yet implemented,
but will eventually be the default for performance reasons. If you require
a stable order, pass `stable=True` for forwards compatibility.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optional name for the operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An int32 `Tensor` with the same shape as `values`. The indices that would
sort each slice of the given `values` along the given `axis`.
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
If axis is not a constant scalar, or the direction is invalid.
</td>
</tr><tr>
<td>
`tf.errors.InvalidArgumentError`
</td>
<td>
If the `values.dtype` is not a `float` or
`int` type.
</td>
</tr>
</table>

