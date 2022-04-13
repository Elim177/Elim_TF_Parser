description: Disable the currently-enabled debugging dumping.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.debugging.experimental.disable_dump_debug_info" />
<meta itemprop="path" content="Stable" />
</div>

# tf.debugging.experimental.disable_dump_debug_info

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/debug/lib/dumping_callback.py">View source</a>



Disable the currently-enabled debugging dumping.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.debugging.experimental.disable_dump_debug_info`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.debugging.experimental.disable_dump_debug_info()
</code></pre>



<!-- Placeholder for "Used in" -->

If the `enable_dump_debug_info()` method under the same Python namespace
has been invoked before, calling this method disables it. If no call to
`enable_dump_debug_info()` has been made, calling this method is a no-op.
Calling this method more than once is idempotent.