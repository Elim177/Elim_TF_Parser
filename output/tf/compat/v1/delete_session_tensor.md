description: Delete the tensor for the given tensor handle.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.delete_session_tensor" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.delete_session_tensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/session_ops.py">View source</a>



Delete the tensor for the given tensor handle.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.delete_session_tensor(
    handle, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is EXPERIMENTAL and subject to change.

Delete the tensor of a given tensor handle. The tensor is produced
in a previous run() and stored in the state of the session.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`handle`
</td>
<td>
The string representation of a persistent tensor handle.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optional name prefix for the return tensor.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A pair of graph elements. The first is a placeholder for feeding a
tensor handle and the second is a deletion operation.
</td>
</tr>

</table>

