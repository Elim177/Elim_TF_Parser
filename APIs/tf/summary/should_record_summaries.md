description: Returns boolean Tensor which is True if summaries will be recorded.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.summary.should_record_summaries" />
<meta itemprop="path" content="Stable" />
</div>

# tf.summary.should_record_summaries

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/summary_ops_v2.py">View source</a>



Returns boolean Tensor which is True if summaries will be recorded.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.summary.should_record_summaries()
</code></pre>



<!-- Placeholder for "Used in" -->

If no default summary writer is currently registered, this always returns
False. Otherwise, this reflects the recording condition has been set via
<a href="../../tf/summary/record_if.md"><code>tf.summary.record_if()</code></a> (except that it may return False for some replicas
when using <a href="../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a>). If no recording condition is active,
it defaults to True.