description: Splits resource variable input tensor across all dimensions.
robots: noindex

# tf.raw_ops.ReadVariableXlaSplitND

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Splits resource variable input tensor across all dimensions.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.ReadVariableXlaSplitND`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.ReadVariableXlaSplitND(
    resource, T, N, num_splits, paddings=[], name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

An op which splits the resource variable input tensor based on the given
num_splits attribute, pads slices optionally, and returned the slices. Slices
are returned in row-major order.

This op may be generated via the TPU bridge.

For example, with `input` tensor:
```
[[0, 1, 2],
 [3, 4, 5],
 [6, 7, 8]]
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
[[0, 1],
 [3, 4]]
[[2, 0],
 [5, 0]]
[[6, 7],
 [0, 0]]
[[8, 0],
 [0, 0]]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`resource`
</td>
<td>
A `Tensor` of type `resource`.
Resource variable of input tensor to split across all dimensions.
  }
  out_arg {
    name: "outputs"
    description: <<END
Output slices based on input and num_splits defined, in row-major order.
</td>
</tr><tr>
<td>
`T`
</td>
<td>
A <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a>.
</td>
</tr><tr>
<td>
`N`
</td>
<td>
An `int` that is `>= 1`.
</td>
</tr><tr>
<td>
`num_splits`
</td>
<td>
A list of `ints`.
Number of ways to split per dimension. Shape dimensions must be evenly
divisible.
</td>
</tr><tr>
<td>
`paddings`
</td>
<td>
An optional list of `ints`. Defaults to `[]`.
Optional list of right paddings per dimension of input tensor to apply before
splitting. This can be used to make a dimension evenly divisible.
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
A list of `N` `Tensor` objects with type `T`.
</td>
</tr>

</table>

