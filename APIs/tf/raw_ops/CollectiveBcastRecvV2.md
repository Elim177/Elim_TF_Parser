description: Receives a tensor value broadcast from another device.
robots: noindex

# tf.raw_ops.CollectiveBcastRecvV2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Receives a tensor value broadcast from another device.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.CollectiveBcastRecvV2`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.CollectiveBcastRecvV2(
    group_size, group_key, instance_key, shape, T,
    communication_hint=&#x27;auto&#x27;, timeout_seconds=0, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`group_size`
</td>
<td>
A `Tensor` of type `int32`.
</td>
</tr><tr>
<td>
`group_key`
</td>
<td>
A `Tensor` of type `int32`.
</td>
</tr><tr>
<td>
`instance_key`
</td>
<td>
A `Tensor` of type `int32`.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`, `int64`.
</td>
</tr><tr>
<td>
`T`
</td>
<td>
A <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a> from: `tf.bool, tf.float32, tf.half, tf.float64, tf.int32, tf.int64`.
</td>
</tr><tr>
<td>
`communication_hint`
</td>
<td>
An optional `string`. Defaults to `"auto"`.
</td>
</tr><tr>
<td>
`timeout_seconds`
</td>
<td>
An optional `float`. Defaults to `0`.
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
A `Tensor` of type `T`.
</td>
</tr>

</table>

