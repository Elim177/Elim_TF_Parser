description: Creates an absolute test srcdir path given a relative path.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.test.test_src_dir_path" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.test.test_src_dir_path

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/platform/test.py">View source</a>



Creates an absolute test srcdir path given a relative path.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.test.test_src_dir_path(
    relative_path
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`relative_path`
</td>
<td>
a path relative to tensorflow root.
e.g. "core/platform".
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An absolute path to the linked in runfiles.
</td>
</tr>

</table>

