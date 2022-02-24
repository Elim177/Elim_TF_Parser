description: Disable filtering out TensorFlow-internal frames in exception stack traces.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.debugging.disable_traceback_filtering" />
<meta itemprop="path" content="Stable" />
</div>

# tf.debugging.disable_traceback_filtering

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/util/traceback_utils.py">View source</a>



Disable filtering out TensorFlow-internal frames in exception stack traces.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.debugging.disable_traceback_filtering`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.debugging.disable_traceback_filtering()
</code></pre>



<!-- Placeholder for "Used in" -->

Raw TensorFlow stack traces involve many internal frames, which can be
challenging to read through, while not being actionable for end users.
By default, TensorFlow filters internal frames in most exceptions that it
raises, to keep stack traces short, readable, and focused on what's
actionable for end users (their own code).

Calling <a href="../../tf/debugging/disable_traceback_filtering.md"><code>tf.debugging.disable_traceback_filtering</code></a> disables this filtering
mechanism, meaning that TensorFlow exceptions stack traces will include
all frames, in particular TensorFlow-internal ones.

**If you are debugging a TensorFlow-internal issue, you need to call
<a href="../../tf/debugging/disable_traceback_filtering.md"><code>tf.debugging.disable_traceback_filtering</code></a>**.
To re-enable traceback filtering afterwards, you can call
<a href="../../tf/debugging/enable_traceback_filtering.md"><code>tf.debugging.enable_traceback_filtering()</code></a>.