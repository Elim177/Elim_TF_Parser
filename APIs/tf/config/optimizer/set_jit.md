description: Configure JIT compilation. (deprecated argument values)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.config.optimizer.set_jit" />
<meta itemprop="path" content="Stable" />
</div>

# tf.config.optimizer.set_jit

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/config.py">View source</a>



Configure JIT compilation. (deprecated argument values)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.config.optimizer.set_jit`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.config.optimizer.set_jit(
    enabled: Union[bool, str]
)
</code></pre>



<!-- Placeholder for "Used in" -->

Warning: SOME ARGUMENT VALUES ARE DEPRECATED: `(jit_config=True)`. They will be removed in a future version.
Instructions for updating:
`True` setting is deprecated, use `autoclustering` instead.

Note: compilation is only applied to code that is compiled into a
graph (in TF2 that's only a code inside <a href="../../../tf/function.md"><code>tf.function</code></a>).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`enabled`
</td>
<td>
JIT compilation configuration.
Possible values:
 - `"autoclustering"` (`True` is a deprecated alias): perform
 [autoclustering](https://www.tensorflow.org/xla#auto-clustering)
   (automatically identify and compile clusters of nodes) on all graphs
   using
 [XLA](https://www.tensorflow.org/xla).
 - `False`: do not automatically compile any graphs.
</td>
</tr>
</table>

