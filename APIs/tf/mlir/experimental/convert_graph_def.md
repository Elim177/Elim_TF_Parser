description: Import a GraphDef and convert it to a textual MLIR module.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.mlir.experimental.convert_graph_def" />
<meta itemprop="path" content="Stable" />
</div>

# tf.mlir.experimental.convert_graph_def

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/compiler/mlir/mlir.py">View source</a>



Import a GraphDef and convert it to a textual MLIR module.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.mlir.experimental.convert_graph_def`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.mlir.experimental.convert_graph_def(
    graph_def, pass_pipeline=&#x27;tf-standard-pipeline&#x27;,
    show_debug_info=(False)
)
</code></pre>



<!-- Placeholder for "Used in" -->

This API is only intended for inspecting the internals of TensorFlow and the
string returned is at the moment intended for debugging purposes.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`graph_def`
</td>
<td>
An object of type graph_pb2.GraphDef or a textual proto
representation of a valid GraphDef.
</td>
</tr><tr>
<td>
`pass_pipeline`
</td>
<td>
A textual description of an MLIR Pass Pipeline to run on the
module, see MLIR documentation for the
[textual pass pipeline syntax](https://mlir.llvm.org/docs/PassManagement/#textual-pass-pipeline-specification).
</td>
</tr><tr>
<td>
`show_debug_info`
</td>
<td>
Whether to include locations in the emitted textual form.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A textual representation of the MLIR module corresponding to the graphdef.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`InvalidArgumentError`
</td>
<td>
if graph_def is invalid or cannot be converted to
MLIR.
</td>
</tr>
</table>

