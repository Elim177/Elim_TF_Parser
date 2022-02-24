description: Check whether traceback filtering is currently enabled.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.debugging.is_traceback_filtering_enabled" />
<meta itemprop="path" content="Stable" />
</div>

# tf.debugging.is_traceback_filtering_enabled

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/util/traceback_utils.py">View source</a>



Check whether traceback filtering is currently enabled.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.debugging.is_traceback_filtering_enabled`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.debugging.is_traceback_filtering_enabled()
</code></pre>



<!-- Placeholder for "Used in" -->

See also <a href="../../tf/debugging/enable_traceback_filtering.md"><code>tf.debugging.enable_traceback_filtering()</code></a> and
<a href="../../tf/debugging/disable_traceback_filtering.md"><code>tf.debugging.disable_traceback_filtering()</code></a>. Note that filtering out
internal frames from the tracebacks of exceptions raised by TensorFlow code
is the default behavior.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
True if traceback filtering is enabled
(e.g. if <a href="../../tf/debugging/enable_traceback_filtering.md"><code>tf.debugging.enable_traceback_filtering()</code></a> was called),
and False otherwise (e.g. if <a href="../../tf/debugging/disable_traceback_filtering.md"><code>tf.debugging.disable_traceback_filtering()</code></a>
was called).
</td>
</tr>

</table>

