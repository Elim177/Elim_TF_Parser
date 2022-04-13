description: Returns JIT compilation configuration for code inside <a href="../../../tf/function.md"><code>tf.function</code></a>.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.config.optimizer.get_jit" />
<meta itemprop="path" content="Stable" />
</div>

# tf.config.optimizer.get_jit

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/config.py">View source</a>



Returns JIT compilation configuration for code inside <a href="../../../tf/function.md"><code>tf.function</code></a>.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.config.optimizer.get_jit`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.config.optimizer.get_jit() -> str
</code></pre>



<!-- Placeholder for "Used in" -->

Possible return values:
   -`"autoclustering"` if
   [autoclustering](https://www.tensorflow.org/xla#auto-clustering) is enabled
   - `""` when no default compilation is applied.