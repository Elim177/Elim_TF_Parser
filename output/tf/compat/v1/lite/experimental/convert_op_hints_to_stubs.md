description: Converts a graphdef with LiteOp hints into stub operations. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.lite.experimental.convert_op_hints_to_stubs" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.lite.experimental.convert_op_hints_to_stubs

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/lite/python/op_hint.py">View source</a>



Converts a graphdef with LiteOp hints into stub operations. (deprecated)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.lite.experimental.convert_op_hints_to_stubs(
    session=None, graph_def=None, write_callback=(lambda graph_def, comments: None)
)
</code></pre>



<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please follow instructions under https://www.tensorflow.org/lite/convert/operation_fusion for operationfusion in tflite.

This is used to prepare for toco conversion of complex intrinsic usages.
Note: only one of session or graph_def should be used, not both.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`session`
</td>
<td>
A TensorFlow session that contains the graph to convert.
</td>
</tr><tr>
<td>
`graph_def`
</td>
<td>
A graph def that we should convert.
</td>
</tr><tr>
<td>
`write_callback`
</td>
<td>
A function pointer that can be used to write intermediate
steps of graph transformation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A new graphdef with all ops contained in OpHints being replaced by
a single op call with the right parameters.
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
If both session and graph_def are provided.
</td>
</tr>
</table>

