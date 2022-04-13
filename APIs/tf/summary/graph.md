description: Writes a TensorFlow graph summary.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.summary.graph" />
<meta itemprop="path" content="Stable" />
</div>

# tf.summary.graph

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/summary_ops_v2.py">View source</a>



Writes a TensorFlow graph summary.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.summary.graph(
    graph_data
)
</code></pre>



<!-- Placeholder for "Used in" -->

Write an instance of <a href="../../tf/Graph.md"><code>tf.Graph</code></a> or <a href="../../tf/compat/v1/GraphDef.md"><code>tf.compat.v1.GraphDef</code></a> as summary only
in an eager mode. Please prefer to use the trace APIs (<a href="../../tf/summary/trace_on.md"><code>tf.summary.trace_on</code></a>,
<a href="../../tf/summary/trace_off.md"><code>tf.summary.trace_off</code></a>, and <a href="../../tf/summary/trace_export.md"><code>tf.summary.trace_export</code></a>) when using
<a href="../../tf/function.md"><code>tf.function</code></a> which can automatically collect and record graphs from
executions.

#### Usage Example:


```py
writer = tf.summary.create_file_writer("/tmp/mylogs")

@tf.function
def f():
  x = constant_op.constant(2)
  y = constant_op.constant(3)
  return x**y

with writer.as_default():
  tf.summary.graph(f.get_concrete_function().graph)

# Another example: in a very rare use case, when you are dealing with a TF v1
# graph.
graph = tf.Graph()
with graph.as_default():
  c = tf.constant(30.0)
with writer.as_default():
  tf.summary.graph(graph)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`graph_data`
</td>
<td>
The TensorFlow graph to write, as a <a href="../../tf/Graph.md"><code>tf.Graph</code></a> or a
<a href="../../tf/compat/v1/GraphDef.md"><code>tf.compat.v1.GraphDef</code></a>.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
True on success, or False if no summary was written because no default
summary writer was available.
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
`graph` summary API is invoked in a graph mode.
</td>
</tr>
</table>

