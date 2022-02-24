description: Forces summary writer to send any buffered data to storage.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.summary.flush" />
<meta itemprop="path" content="Stable" />
</div>

# tf.summary.flush

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/summary_ops_v2.py">View source</a>



Forces summary writer to send any buffered data to storage.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.summary.flush(
    writer=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This operation blocks until that finishes.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`writer`
</td>
<td>
The <a href="../../tf/summary/SummaryWriter.md"><code>tf.summary.SummaryWriter</code></a> to flush. If None, the current
default writer will be used instead; if there is no current writer, this
returns <a href="../../tf/no_op.md"><code>tf.no_op</code></a>.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Ignored legacy argument for a name for the operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The created <a href="../../tf/Operation.md"><code>tf.Operation</code></a>.
</td>
</tr>

</table>

