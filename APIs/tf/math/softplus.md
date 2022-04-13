description: Computes elementwise softplus: softplus(x) = log(exp(x) + 1).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.softplus" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.softplus

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>



Computes elementwise softplus: `softplus(x) = log(exp(x) + 1)`.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.nn.softplus`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.math.softplus`, `tf.compat.v1.nn.softplus`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.softplus(
    features, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

`softplus` is a smooth approximation of `relu`. Like `relu`, `softplus` always
takes on positive values.

<img style="width:100%" src="https://www.tensorflow.org/images/softplus.png">

#### Example:



```
>>> import tensorflow as tf
>>> tf.math.softplus(tf.range(0, 2, dtype=tf.float32)).numpy()
array([0.6931472, 1.3132616], dtype=float32)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`features`
</td>
<td>
`Tensor`
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optional: name to associate with this operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
`Tensor`
</td>
</tr>

</table>

