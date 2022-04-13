description: Represents options for dataset optimizations.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.OptimizationOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
</div>

# tf.data.experimental.OptimizationOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/data/ops/options.py">View source</a>



Represents options for dataset optimizations.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.data.experimental.OptimizationOptions`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.OptimizationOptions()
</code></pre>



<!-- Placeholder for "Used in" -->

You can set the optimization options of a dataset through the
`experimental_optimization` property of <a href="../../../tf/data/Options.md"><code>tf.data.Options</code></a>; the property is
an instance of <a href="../../../tf/data/experimental/OptimizationOptions.md"><code>tf.data.experimental.OptimizationOptions</code></a>.

```python
options = tf.data.Options()
options.experimental_optimization.noop_elimination = True
options.experimental_optimization.apply_default_optimizations = False
dataset = dataset.with_options(options)
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`apply_default_optimizations`
</td>
<td>
Whether to apply default graph optimizations. If False, only graph optimizations that have been explicitly enabled will be applied.
</td>
</tr><tr>
<td>
`filter_fusion`
</td>
<td>
Whether to fuse filter transformations. If None, defaults to False.
</td>
</tr><tr>
<td>
`map_and_batch_fusion`
</td>
<td>
Whether to fuse map and batch transformations. If None, defaults to True.
</td>
</tr><tr>
<td>
`map_and_filter_fusion`
</td>
<td>
Whether to fuse map and filter transformations. If None, defaults to False.
</td>
</tr><tr>
<td>
`map_fusion`
</td>
<td>
Whether to fuse map transformations. If None, defaults to False.
</td>
</tr><tr>
<td>
`map_parallelization`
</td>
<td>
Whether to parallelize stateless map transformations. If None, defaults to True.
</td>
</tr><tr>
<td>
`noop_elimination`
</td>
<td>
Whether to eliminate no-op transformations. If None, defaults to True.
</td>
</tr><tr>
<td>
`parallel_batch`
</td>
<td>
Whether to parallelize copying of batch elements. This optimization is highly experimental and can cause performance degradation (e.g. when the parallelization overhead exceeds the benefits of performing the data copies in parallel). You should only enable this optimization if a) your input pipeline is bottlenecked on batching and b) you have validated that this optimization improves performance. If None, defaults to False.
</td>
</tr><tr>
<td>
`shuffle_and_repeat_fusion`
</td>
<td>
Whether to fuse shuffle and repeat transformations. If None, defaults to True.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/data/util/options.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.


<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/data/util/options.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Return self!=value.




