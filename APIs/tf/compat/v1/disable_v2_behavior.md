description: Disables TensorFlow 2.x behaviors.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.disable_v2_behavior" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.disable_v2_behavior

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/compat/v2_compat.py">View source</a>



Disables TensorFlow 2.x behaviors.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.disable_v2_behavior()
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

Using this function indicates that your software is not compatible
with eager execution and <a href="../../../tf/function.md"><code>tf.function</code></a> in TF2.

To migrate to TF2, rewrite your code to be compatible with eager execution.
Please refer to the [migration guide]
(https://www.tensorflow.org/guide/migrate) for additional resource on the
topic.


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

This function can be called at the beginning of the program (before `Tensors`,
`Graphs` or other structures have been created, and before devices have been
initialized. It switches all global behaviors that are different between
TensorFlow 1.x and 2.x to behave as intended for 1.x.

User can call this function to disable 2.x behavior during complex migrations.

