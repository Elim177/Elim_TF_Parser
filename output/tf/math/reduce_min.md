description: Computes the <a href="../../tf/math/minimum.md"><code>tf.math.minimum</code></a> of elements across dimensions of a tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.reduce_min" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.reduce_min

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>



Computes the <a href="../../tf/math/minimum.md"><code>tf.math.minimum</code></a> of elements across dimensions of a tensor.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.reduce_min`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.reduce_min(
    input_tensor, axis=None, keepdims=(False), name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is the reduction operation for the elementwise <a href="../../tf/math/minimum.md"><code>tf.math.minimum</code></a> op.

Reduces `input_tensor` along the dimensions given in `axis`.
Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
of the entries in `axis`, which must be unique. If `keepdims` is true, the
reduced dimensions are retained with length 1.

If `axis` is None, all dimensions are reduced, and a
tensor with a single element is returned.

#### For example:



```
>>> a = tf.constant([
...   [[1, 2], [3, 4]],
...   [[1, 2], [3, 4]]
... ])
>>> tf.reduce_min(a)
<tf.Tensor: shape=(), dtype=int32, numpy=1>
```

Choosing a specific axis returns minimum element in the given axis:

```
>>> b = tf.constant([[1, 2, 3], [4, 5, 6]])
>>> tf.reduce_min(b, axis=0)
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>
>>> tf.reduce_min(b, axis=1)
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 4], dtype=int32)>
```

Setting `keepdims` to `True` retains the dimension of `input_tensor`:

```
>>> tf.reduce_min(a, keepdims=True)
<tf.Tensor: shape=(1, 1, 1), dtype=int32, numpy=array([[[1]]], dtype=int32)>
>>> tf.math.reduce_min(a, axis=0, keepdims=True)
<tf.Tensor: shape=(1, 2, 2), dtype=int32, numpy=
array([[[1, 2],
        [3, 4]]], dtype=int32)>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_tensor`
</td>
<td>
The tensor to reduce. Should have real numeric type.
</td>
</tr><tr>
<td>
`axis`
</td>
<td>
The dimensions to reduce. If `None` (the default), reduces all
dimensions. Must be in the range `[-rank(input_tensor),
rank(input_tensor))`.
</td>
</tr><tr>
<td>
`keepdims`
</td>
<td>
If true, retains reduced dimensions with length 1.
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
The reduced tensor.
</td>
</tr>

</table>




 <section><devsite-expandable expanded>
 <h2 class="showalways">numpy compatibility</h2>

Equivalent to np.min


 </devsite-expandable></section>

