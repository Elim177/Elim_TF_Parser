description: Opts out of control flow v2.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.disable_control_flow_v2" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.disable_control_flow_v2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/control_flow_v2_toggles.py">View source</a>



Opts out of control flow v2.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.disable_control_flow_v2()
</code></pre>



<!-- Placeholder for "Used in" -->

Note: v2 control flow is always enabled inside of tf.function. Calling this
function has no effect in that case.

If your code needs tf.disable_control_flow_v2() to be called to work
properly please file a bug.