description: Searches for where a value would go in a sorted sequence.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.searchsorted" />
<meta itemprop="path" content="Stable" />
</div>

# tf.searchsorted

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/array_ops.py">View source</a>



Searches for where a value would go in a sorted sequence.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.searchsorted`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.searchsorted(
    sorted_sequence, values, side=&#x27;left&#x27;, out_type=tf.dtypes.int32,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is not a method for checking containment (like python `in`).

The typical use case for this operation is "binning", "bucketing", or
"discretizing". The `values` are assigned to bucket-indices based on the
**edges** listed in `sorted_sequence`. This operation
returns the bucket-index for each value.

```
>>> edges = [-1, 3.3, 9.1, 10.0]
>>> values = [0.0, 4.1, 12.0]
>>> tf.searchsorted(edges, values).numpy()
array([1, 2, 4], dtype=int32)
```

The `side` argument controls which index is returned if a value lands exactly
on an edge:

```
>>> seq = [0, 3, 9, 10, 10]
>>> values = [0, 4, 10]
>>> tf.searchsorted(seq, values).numpy()
array([0, 2, 3], dtype=int32)
>>> tf.searchsorted(seq, values, side="right").numpy()
array([1, 2, 5], dtype=int32)
```

The `axis` is not settable for this operation. It always operates on the
innermost dimension (`axis=-1`). The operation will accept any number of
outer dimensions. Here it is applied to the rows of a matrix:

```
>>> sorted_sequence = [[0., 3., 8., 9., 10.],
...                    [1., 2., 3., 4., 5.]]
>>> values = [[9.8, 2.1, 4.3],
...           [0.1, 6.6, 4.5, ]]
>>> tf.searchsorted(sorted_sequence, values).numpy()
array([[4, 1, 2],
       [0, 5, 4]], dtype=int32)
```

Note: This operation assumes that `sorted_sequence` **is sorted** along the
innermost axis, maybe using `tf.sort(..., axis=-1)`. **If the sequence is not
sorted no error is raised** and the content of the returned tensor is not well
defined.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sorted_sequence`
</td>
<td>
N-D `Tensor` containing a sorted sequence.
</td>
</tr><tr>
<td>
`values`
</td>
<td>
N-D `Tensor` containing the search values.
</td>
</tr><tr>
<td>
`side`
</td>
<td>
'left' or 'right'; 'left' corresponds to lower_bound and 'right' to
upper_bound.
</td>
</tr><tr>
<td>
`out_type`
</td>
<td>
The output type (`int32` or `int64`).  Default is <a href="../tf.md#int32"><code>tf.int32</code></a>.
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
An N-D `Tensor` the size of `values` containing the result of applying
either lower_bound or upper_bound (depending on side) to each value.  The
result is not a global index to the entire `Tensor`, but the index in the
last dimension.
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
If the last dimension of `sorted_sequence >= 2^31-1` elements.
If the total size of `values` exceeds `2^31 - 1` elements.
If the first `N-1` dimensions of the two tensors don't match.
</td>
</tr>
</table>

