description: Prunes out nodes that aren't needed for inference. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.graph_util.remove_training_nodes" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.graph_util.remove_training_nodes

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/graph_util_impl.py">View source</a>



Prunes out nodes that aren't needed for inference. (deprecated)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.graph_util.remove_training_nodes(
    input_graph, protected_nodes=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use <a href="../../../../tf/compat/v1/graph_util/remove_training_nodes.md"><code>tf.compat.v1.graph_util.remove_training_nodes</code></a>

There are nodes like Identity and CheckNumerics that are only useful
during training, and can be removed in graphs that will be used for
nothing but inference. Here we identify and remove them, returning an
equivalent graph. To be specific, CheckNumerics nodes are always removed, and
Identity nodes that aren't involved in control edges are spliced out so that
their input and outputs are directly connected.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_graph`
</td>
<td>
Model to analyze and prune.
</td>
</tr><tr>
<td>
`protected_nodes`
</td>
<td>
An optional list of names of nodes to be kept
unconditionally. This is for example useful to preserve Identity output
nodes.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of nodes with the unnecessary ones removed.
</td>
</tr>

</table>

