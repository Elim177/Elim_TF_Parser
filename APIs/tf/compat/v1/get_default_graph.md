description: Returns the default graph for the current thread.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.get_default_graph" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.get_default_graph

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>



Returns the default graph for the current thread.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.get_default_graph()
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

`get_default_graph` does not work with either eager execution or
<a href="../../../tf/function.md"><code>tf.function</code></a>, and you should not invoke it directly. To migrate code that
uses Graph-related functions to TF2, rewrite the code without them. See the
[migration guide](https://www.tensorflow.org/guide/migrate) for more
description about the behavior and semantic changes between Tensorflow 1 and
Tensorflow 2.


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

The returned graph will be the innermost graph on which a
`Graph.as_default()` context has been entered, or a global default
graph if none has been explicitly created.

NOTE: The default graph is a property of the current thread. If you
create a new thread, and wish to use the default graph in that
thread, you must explicitly add a `with g.as_default():` in that
thread's function.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The default `Graph` being used in the current thread.
</td>
</tr>

</table>

