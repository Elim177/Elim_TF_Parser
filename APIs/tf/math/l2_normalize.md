description: Normalizes along dimension axis using an L2 norm. (deprecated arguments)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.l2_normalize" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.l2_normalize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/nn_impl.py">View source</a>



Normalizes along dimension `axis` using an L2 norm. (deprecated arguments)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.linalg.l2_normalize`, `tf.nn.l2_normalize`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.linalg.l2_normalize`, `tf.compat.v1.math.l2_normalize`, `tf.compat.v1.nn.l2_normalize`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.l2_normalize(
    x, axis=None, epsilon=1e-12, name=None, dim=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Warning: SOME ARGUMENTS ARE DEPRECATED: `(dim)`. They will be removed in a future version.
Instructions for updating:
dim is deprecated, use axis instead

For a 1-D tensor with `axis = 0`, computes

    output = x / sqrt(max(sum(x**2), epsilon))

For `x` with more dimensions, independently normalizes each 1-D slice along
dimension `axis`.

1-D tensor example:
```
>>> x = tf.constant([3.0, 4.0])
>>> tf.math.l2_normalize(x).numpy()
array([0.6, 0.8], dtype=float32)
```

2-D tensor example:
```
>>> x = tf.constant([[3.0], [4.0]])
>>> tf.math.l2_normalize(x, 0).numpy()
array([[0.6],
     [0.8]], dtype=float32)
```

```
>>> x = tf.constant([[3.0], [4.0]])
>>> tf.math.l2_normalize(x, 1).numpy()
array([[1.],
     [1.]], dtype=float32)
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
A `Tensor`.
</td>
</tr><tr>
<td>
`axis`
</td>
<td>
Dimension along which to normalize.  A scalar or a vector of
integers.
</td>
</tr><tr>
<td>
`epsilon`
</td>
<td>
A lower bound value for the norm. Will use `sqrt(epsilon)` as the
divisor if `norm < sqrt(epsilon)`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for this operation (optional).
</td>
</tr><tr>
<td>
`dim`
</td>
<td>
Deprecated, do not use.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` with the same shape as `x`.
</td>
</tr>

</table>

