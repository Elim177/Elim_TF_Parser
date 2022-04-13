description: Converts the given type_value to a DType.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.dtypes.as_dtype" />
<meta itemprop="path" content="Stable" />
</div>

# tf.dtypes.as_dtype

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/dtypes.py">View source</a>



Converts the given `type_value` to a `DType`.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.as_dtype`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.as_dtype`, `tf.compat.v1.dtypes.as_dtype`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.dtypes.as_dtype(
    type_value
)
</code></pre>



<!-- Placeholder for "Used in" -->

Note: `DType` values are interned. When passed a new `DType` object,
`as_dtype` always returns the interned value.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`type_value`
</td>
<td>
A value that can be converted to a <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a> object. This may
currently be a <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a> object, a [`DataType`
enum](https://www.tensorflow.org/code/tensorflow/core/framework/types.proto),
  a string type name, or a [`numpy.dtype`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `DType` corresponding to `type_value`.
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
If `type_value` cannot be converted to a `DType`.
</td>
</tr>
</table>

