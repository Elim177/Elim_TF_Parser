description: Sets the default summary step for the current thread.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.summary.experimental.set_step" />
<meta itemprop="path" content="Stable" />
</div>

# tf.summary.experimental.set_step

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/summary_ops_v2.py">View source</a>



Sets the default summary step for the current thread.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.summary.experimental.set_step(
    step
)
</code></pre>



<!-- Placeholder for "Used in" -->

For convenience, this function sets a default value for the `step` parameter
used in summary-writing functions elsewhere in the API so that it need not
be explicitly passed in every such invocation. The value can be a constant
or a variable, and can be retrieved via <a href="../../../tf/summary/experimental/get_step.md"><code>tf.summary.experimental.get_step()</code></a>.

Note: when using this with @tf.functions, the step value will be captured at
the time the function is traced, so changes to the step outside the function
will not be reflected inside the function unless using a <a href="../../../tf/Variable.md"><code>tf.Variable</code></a> step.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`step`
</td>
<td>
An `int64`-castable default step value, or None to unset.
</td>
</tr>
</table>

