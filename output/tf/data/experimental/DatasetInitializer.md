description: Creates a table initializer from a <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a>.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.DatasetInitializer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initialize"/>
</div>

# tf.data.experimental.DatasetInitializer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/data/experimental/ops/lookup_ops.py">View source</a>



Creates a table initializer from a <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a>.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.data.experimental.DatasetInitializer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.DatasetInitializer(
    dataset
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Sample usage:



```
>>> keys = tf.data.Dataset.range(100)
>>> values = tf.data.Dataset.range(100).map(
...     lambda x: tf.strings.as_string(x * 2))
>>> ds = tf.data.Dataset.zip((keys, values))
>>> init = tf.data.experimental.DatasetInitializer(ds)
>>> table = tf.lookup.StaticHashTable(init, "")
>>> table.lookup(tf.constant([0, 1, 2], dtype=tf.int64)).numpy()
array([b'0', b'2', b'4'], dtype=object)
```
Raises: ValueError if `dataset` doesn't conform to specifications.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dataset`
</td>
<td>
A <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> object that produces tuples of scalars. The
first scalar is treated as a key and the second as value.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`dataset`
</td>
<td>
A <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> object that produces tuples of scalars. The
first scalar is treated as a key and the second as value.
</td>
</tr><tr>
<td>
`key_dtype`
</td>
<td>
The expected table key dtype.
</td>
</tr><tr>
<td>
`value_dtype`
</td>
<td>
The expected table value dtype.
</td>
</tr>
</table>



## Methods

<h3 id="initialize"><code>initialize</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/data/experimental/ops/lookup_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>initialize(
    table
)
</code></pre>

Returns the table initialization op.




