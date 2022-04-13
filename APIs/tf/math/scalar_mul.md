description: Multiplies a scalar times a Tensor or IndexedSlices object.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.scalar_mul" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.scalar_mul

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>



Multiplies a scalar times a `Tensor` or `IndexedSlices` object.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.scalar_mul`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.scalar_mul(
    scalar, x, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is a special case of <a href="../../tf/math/multiply.md"><code>tf.math.multiply</code></a>, where the first value must be a
`scalar`. Unlike the general form of <a href="../../tf/math/multiply.md"><code>tf.math.multiply</code></a>, this is operation is
guaranteed to be efficient for <a href="../../tf/IndexedSlices.md"><code>tf.IndexedSlices</code></a>.

```
>>> x = tf.reshape(tf.range(30, dtype=tf.float32), [10, 3])
>>> with tf.GradientTape() as g:
...   g.watch(x)
...   y = tf.gather(x, [1, 2])  # IndexedSlices
...   z = tf.math.scalar_mul(10.0, y)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`scalar`
</td>
<td>
A 0-D scalar `Tensor`. Must have known shape.
</td>
</tr><tr>
<td>
`x`
</td>
<td>
A `Tensor` or `IndexedSlices` to be scaled.
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
`scalar * x` of the same type (`Tensor` or `IndexedSlices`) as `x`.
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
if scalar is not a 0-D `scalar`.
</td>
</tr>
</table>

