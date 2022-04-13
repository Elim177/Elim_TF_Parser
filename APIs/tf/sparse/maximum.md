description: Returns the element-wise max of two SparseTensors.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.sparse.maximum" />
<meta itemprop="path" content="Stable" />
</div>

# tf.sparse.maximum

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/sparse_ops.py">View source</a>



Returns the element-wise max of two SparseTensors.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.sparse.maximum`, `tf.compat.v1.sparse_maximum`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.sparse.maximum(
    sp_a, sp_b, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Assumes the two SparseTensors have the same shape, i.e., no broadcasting.

#### Example:


```
>>> sp_zero = tf.sparse.SparseTensor([[0]], [0], [7])
>>> sp_one = tf.sparse.SparseTensor([[1]], [1], [7])
>>> res = tf.sparse.maximum(sp_zero, sp_one)
>>> res.indices
<tf.Tensor: shape=(2, 1), dtype=int64, numpy=
array([[0],
       [1]])>
>>> res.values
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([0, 1], dtype=int32)>
>>> res.dense_shape
<tf.Tensor: shape=(1,), dtype=int64, numpy=array([7])>
```


The reduction version of this elementwise operation is <a href="../../tf/sparse/reduce_max.md"><code>tf.sparse.reduce_max</code></a>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sp_a`
</td>
<td>
a `SparseTensor` operand whose dtype is real, and indices
lexicographically ordered.
</td>
</tr><tr>
<td>
`sp_b`
</td>
<td>
the other `SparseTensor` operand with the same requirements (and the
same shape).
</td>
</tr><tr>
<td>
`name`
</td>
<td>
optional name of the operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`output`
</td>
<td>
the output SparseTensor.
</td>
</tr>
</table>

