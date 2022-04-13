description: Represents options for dataset threading.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.ThreadingOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
</div>

# tf.data.ThreadingOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/data/ops/options.py">View source</a>



Represents options for dataset threading.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.data.experimental.ThreadingOptions`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.data.ThreadingOptions`, `tf.compat.v1.data.experimental.ThreadingOptions`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.ThreadingOptions()
</code></pre>



<!-- Placeholder for "Used in" -->

You can set the threading options of a dataset through the
`experimental_threading` property of <a href="../../tf/data/Options.md"><code>tf.data.Options</code></a>; the property is
an instance of <a href="../../tf/data/ThreadingOptions.md"><code>tf.data.ThreadingOptions</code></a>.

```python
options = tf.data.Options()
options.threading.private_threadpool_size = 10
dataset = dataset.with_options(options)
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`max_intra_op_parallelism`
</td>
<td>
If set, it overrides the maximum degree of intra-op parallelism.
</td>
</tr><tr>
<td>
`private_threadpool_size`
</td>
<td>
If set, the dataset will use a private threadpool of the given size. The value 0 can be used to indicate that the threadpool size should be determined at runtime based on the number of available CPU cores.
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




