description: Cache for file writers.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.summary.FileWriterCache" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="clear"/>
<meta itemprop="property" content="get"/>
</div>

# tf.compat.v1.summary.FileWriterCache

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/summary/writer/writer_cache.py">View source</a>



Cache for file writers.

<!-- Placeholder for "Used in" -->

This class caches file writers, one per directory.

## Methods

<h3 id="clear"><code>clear</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/summary/writer/writer_cache.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>clear()
</code></pre>

Clear cached summary writers. Currently only used for unit tests.


<h3 id="get"><code>get</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/summary/writer/writer_cache.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>get(
    logdir
)
</code></pre>

Returns the FileWriter for the specified directory.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`logdir`
</td>
<td>
str, name of the directory.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `FileWriter`.
</td>
</tr>

</table>





