description: XLA compilation options.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.tpu.XLAOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="enable_xla_dynamic_padder"/>
<meta itemprop="property" content="use_spmd_for_xla_partitioning"/>
</div>

# tf.tpu.XLAOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/tpu/tpu.py">View source</a>



XLA compilation options.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.tpu.XLAOptions`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.tpu.XLAOptions(
    use_spmd_for_xla_partitioning=(True), enable_xla_dynamic_padder=(True)
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`use_spmd_for_xla_partitioning`
</td>
<td>
Boolean. Whether to use XLA's SPMD
partitioner instead of MPMD partitioner when compiler partitioning is
requested.
</td>
</tr><tr>
<td>
`enable_xla_dynamic_padder`
</td>
<td>
Boolean. Whether to enable XLA dynamic padder
infrastructure to handle dynamic shapes inputs inside XLA. True by
default. Disabling this may cause correctness issues with dynamic shapes
inputs, as XLA will just assume the inputs are with padded shapes. However
users can optionally set it to False to improve device time if masking is
already handled in the user side.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
enable_xla_dynamic_padder<a id="enable_xla_dynamic_padder"></a>
</td>
<td>
Instance of `_collections._tuplegetter`

Alias for field number 1
</td>
</tr><tr>
<td>
use_spmd_for_xla_partitioning<a id="use_spmd_for_xla_partitioning"></a>
</td>
<td>
Instance of `_collections._tuplegetter`

Alias for field number 0
</td>
</tr>
</table>

