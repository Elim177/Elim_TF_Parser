description: Computes the minimum along segments of a tensor.
robots: noindex

# tf.raw_ops.SegmentMin

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Computes the minimum along segments of a tensor.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.SegmentMin`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.SegmentMin(
    data, segment_ids, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \min_j(data_j)\\) where `min` is over `j` such
that `segment_ids[j] == i`.

If the min is empty for a given segment ID `i`, `output[i] = 0`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/SegmentMin.png" alt>
</div>

#### For example:



```
c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
tf.segment_min(c, tf.constant([0, 0, 1]))
# ==> [[1, 2, 2, 1],
#      [5, 6, 7, 8]]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`data`
</td>
<td>
A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
</td>
</tr><tr>
<td>
`segment_ids`
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`, `int64`.
A 1-D tensor whose size is equal to the size of `data`'s
first dimension.  Values should be sorted and can be repeated.
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
A `Tensor`. Has the same type as `data`.
</td>
</tr>

</table>
