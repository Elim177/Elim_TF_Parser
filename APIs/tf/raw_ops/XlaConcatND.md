description: Concats input tensor across all dimensions.
robots: noindex

# tf.raw_ops.XlaConcatND

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Concats input tensor across all dimensions.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.XlaConcatND`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.XlaConcatND(
    inputs, num_concats, paddings=[], name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

An op which merges slices the input tensor based on the given num_splits
attribute, strips paddings optionally, and returns the merged tensor without
paddings.

This op may be generated via the TPU bridge.

For example, with `input` tensor:
```
[[0, 1],
 [4, 5]]
[[2, 3],
 [6, 7]]
[[8, 9],
 [12, 13]]
[[10, 11],
 [14, 15]]
```
`num_splits`:
```
[2, 2]
```
and `paddings`:
```
[1, 1]
```
the expected `outputs` is:
```
[[0, 1, 2],
 [4, 5, 6],
 [8, 9, 10]]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`inputs`
</td>
<td>
A list of at least 1 `Tensor` objects with the same type.
Input tensor slices in row-major order to merge across all dimensions. All
inputs must have the same shape.
  }
  out_arg {
    name: "output"
    description: <<END
Output tensor formed from merging input slices based on num_concats defined.
</td>
</tr><tr>
<td>
`num_concats`
</td>
<td>
A list of `ints`. Number of ways to merge per dimension.
</td>
</tr><tr>
<td>
`paddings`
</td>
<td>
An optional list of `ints`. Defaults to `[]`.
Optional list of right paddings per dimension to strip from the final merged
tensor. These paddings must not exceed the dimension size of the merged result
prior to stripping paddings.
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
A `Tensor`. Has the same type as `inputs`.
</td>
</tr>

</table>

