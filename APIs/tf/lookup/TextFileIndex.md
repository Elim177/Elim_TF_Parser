description: The key and value content to get from each line.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.lookup.TextFileIndex" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="LINE_NUMBER"/>
<meta itemprop="property" content="WHOLE_LINE"/>
</div>

# tf.lookup.TextFileIndex

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/lookup_ops.py">View source</a>



The key and value content to get from each line.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.lookup.TextFileIndex`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

This class defines the key and value used for <a href="../../tf/lookup/TextFileInitializer.md"><code>tf.lookup.TextFileInitializer</code></a>.

The key and value content to get from each line is specified either
by the following, or a value `>=0`.
* <a href="../../tf/lookup/TextFileIndex.md#LINE_NUMBER"><code>TextFileIndex.LINE_NUMBER</code></a> means use the line number starting from zero,
  expects data type int64.
* <a href="../../tf/lookup/TextFileIndex.md#WHOLE_LINE"><code>TextFileIndex.WHOLE_LINE</code></a> means use the whole line content, expects data
  type string.

A value `>=0` means use the index (starting at zero) of the split line based
    on `delimiter`.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
LINE_NUMBER<a id="LINE_NUMBER"></a>
</td>
<td>
`-1`
</td>
</tr><tr>
<td>
WHOLE_LINE<a id="WHOLE_LINE"></a>
</td>
<td>
`-2`
</td>
</tr>
</table>

