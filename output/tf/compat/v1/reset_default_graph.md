description: Clears the default graph stack and resets the global default graph.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.reset_default_graph" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.reset_default_graph

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>



Clears the default graph stack and resets the global default graph.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.reset_default_graph()
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

`reset_default_graph` does not work with either eager execution or
<a href="../../../tf/function.md"><code>tf.function</code></a>, and you should not invoke it directly. To migrate code that
uses Graph-related functions to TF2, rewrite the code without them. See the
[migration guide](https://www.tensorflow.org/guide/migrate) for more
description about the behavior and semantic changes between Tensorflow 1 and
Tensorflow 2.


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

NOTE: The default graph is a property of the current thread. This
function applies only to the current thread.  Calling this function while
a <a href="../../../tf/compat/v1/Session.md"><code>tf.compat.v1.Session</code></a> or <a href="../../../tf/compat/v1/InteractiveSession.md"><code>tf.compat.v1.InteractiveSession</code></a> is active will
result in undefined
behavior. Using any previously created <a href="../../../tf/Operation.md"><code>tf.Operation</code></a> or <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a> objects
after calling this function will result in undefined behavior.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`AssertionError`
</td>
<td>
If this function is called within a nested graph.
</td>
</tr>
</table>

