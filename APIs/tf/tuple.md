description: Groups tensors together.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.tuple" />
<meta itemprop="path" content="Stable" />
</div>

# tf.tuple

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/control_flow_ops.py">View source</a>



Groups tensors together.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.tuple(
    tensors, control_inputs=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The returned tensors have the same value as the input tensors, but they
are computed only after all the input tensors have been computed.

Note: *In TensorFlow 2 with eager and/or Autograph, you should not require
this method, as ops execute in the expected order thanks to automatic control
dependencies.* Only use <a href="../tf/tuple.md"><code>tf.tuple</code></a> when working with v1 <a href="../tf/Graph.md"><code>tf.Graph</code></a> code.

See also <a href="../tf/group.md"><code>tf.group</code></a> and <a href="../tf/control_dependencies.md"><code>tf.control_dependencies</code></a>.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensors`
</td>
<td>
A list of `Tensor`s or `IndexedSlices`, some entries can be `None`.
</td>
</tr><tr>
<td>
`control_inputs`
</td>
<td>
List of additional ops to finish before returning.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
(optional) A name to use as a `name_scope` for the operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Same as `tensors`.
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
If `tensors` does not contain any `Tensor` or `IndexedSlices`.
</td>
</tr><tr>
<td>
`TypeError`
</td>
<td>
If `control_inputs` is not a list of `Operation` or `Tensor`
objects.
</td>
</tr>
</table>

