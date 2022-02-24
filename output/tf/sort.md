description: Sorts a tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.sort" />
<meta itemprop="path" content="Stable" />
</div>

# tf.sort

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/sort_ops.py">View source</a>



Sorts a tensor.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.sort`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.sort(
    values, axis=-1, direction=&#x27;ASCENDING&#x27;, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Usage:



```
>>> a = [1, 10, 26.9, 2.8, 166.32, 62.3]
>>> tf.sort(a).numpy()
array([  1.  ,   2.8 ,  10.  ,  26.9 ,  62.3 , 166.32], dtype=float32)
```

```
>>> tf.sort(a, direction='DESCENDING').numpy()
array([166.32,  62.3 ,  26.9 ,  10.  ,   2.8 ,   1.  ], dtype=float32)
```

For multidimensional inputs you can control which axis the sort is applied
along. The default `axis=-1` sorts the innermost axis.

```
>>> mat = [[3,2,1],
...        [2,1,3],
...        [1,3,2]]
>>> tf.sort(mat, axis=-1).numpy()
array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]], dtype=int32)
>>> tf.sort(mat, axis=0).numpy()
array([[1, 1, 1],
       [2, 2, 2],
       [3, 3, 3]], dtype=int32)
```

#### See also:


* <a href="../tf/argsort.md"><code>tf.argsort</code></a>: Like sort, but it returns the sort indices.
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
A `Tensor` with the same dtype and shape as `values`, with the elements
sorted along the given `axis`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`tf.errors.InvalidArgumentError`
</td>
<td>
If the `values.dtype` is not a `float` or
`int` type.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If axis is not a constant scalar, or the direction is invalid.
</td>
</tr>
</table>

