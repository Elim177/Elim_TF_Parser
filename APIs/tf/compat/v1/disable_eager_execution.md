description: Disables eager execution.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.disable_eager_execution" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.disable_eager_execution

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>



Disables eager execution.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.disable_eager_execution()
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

This function is not necessary if you are using TF2. Eager execution is
enabled by default. If you want to use Graph mode please consider
[tf.function](https://www.tensorflow.org/api_docs/python/tf/function).


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

This function can only be called before any Graphs, Ops, or Tensors have been
created.

