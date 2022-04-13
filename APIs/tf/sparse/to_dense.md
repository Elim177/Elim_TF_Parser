description: Converts a SparseTensor into a dense tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.sparse.to_dense" />
<meta itemprop="path" content="Stable" />
</div>

# tf.sparse.to_dense

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/sparse_ops.py">View source</a>



Converts a `SparseTensor` into a dense tensor.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.sparse.to_dense`, `tf.compat.v1.sparse_tensor_to_dense`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.sparse.to_dense(
    sp_input, default_value=None, validate_indices=(True), name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

For this sparse tensor with three non-empty values:

```
>>> sp_input = tf.SparseTensor(
...   dense_shape=[3, 5],
...   values=[7, 8, 9],
...   indices =[[0, 1],
...             [0, 3],
...             [2, 0]])
```

The output will be a dense `[3, 5]` tensor with values:

```
>>> tf.sparse.to_dense(sp_input).numpy()
array([[0, 7, 0, 8, 0],
       [0, 0, 0, 0, 0],
       [9, 0, 0, 0, 0]], dtype=int32)
```

Note: Indices must be without repeats.  This is only tested if
`validate_indices` is `True`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sp_input`
</td>
<td>
The input `SparseTensor`.
</td>
</tr><tr>
<td>
`default_value`
</td>
<td>
Scalar value to set for indices not specified in
`sp_input`.  Defaults to zero.
</td>
</tr><tr>
<td>
`validate_indices`
</td>
<td>
A boolean value.  If `True`, indices are checked to make
sure they are sorted in lexicographic order and that there are no repeats.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name prefix for the returned tensors (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dense tensor with shape `sp_input.dense_shape` and values specified by
the non-empty values in `sp_input`. Indices not in `sp_input` are assigned
`default_value`.
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
If `sp_input` is not a `SparseTensor`.
</td>
</tr>
</table>

