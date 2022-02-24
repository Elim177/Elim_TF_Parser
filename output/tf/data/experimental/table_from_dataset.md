description: Returns a lookup table based on the given dataset.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.table_from_dataset" />
<meta itemprop="path" content="Stable" />
</div>

# tf.data.experimental.table_from_dataset

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/data/experimental/ops/lookup_ops.py">View source</a>



Returns a lookup table based on the given dataset.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.data.experimental.table_from_dataset`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.table_from_dataset(
    dataset=None, num_oov_buckets=0, vocab_size=None, default_value=None,
    hasher_spec=lookup_ops.FastHashSpec, key_dtype=tf.dtypes.string, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This operation constructs a lookup table based on the given dataset of pairs
of (key, value).

Any lookup of an out-of-vocabulary token will return a bucket ID based on its
hash if `num_oov_buckets` is greater than zero. Otherwise it is assigned the
`default_value`.
The bucket ID range is
`[vocabulary size, vocabulary size + num_oov_buckets - 1]`.

#### Sample Usages:



```
>>> keys = tf.data.Dataset.range(100)
>>> values = tf.data.Dataset.range(100).map(
...     lambda x: tf.strings.as_string(x * 2))
>>> ds = tf.data.Dataset.zip((keys, values))
>>> table = tf.data.experimental.table_from_dataset(
...                               ds, default_value='n/a', key_dtype=tf.int64)
>>> table.lookup(tf.constant([0, 1, 2], dtype=tf.int64)).numpy()
array([b'0', b'2', b'4'], dtype=object)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dataset`
</td>
<td>
A dataset containing (key, value) pairs.
</td>
</tr><tr>
<td>
`num_oov_buckets`
</td>
<td>
The number of out-of-vocabulary buckets.
</td>
</tr><tr>
<td>
`vocab_size`
</td>
<td>
Number of the elements in the vocabulary, if known.
</td>
</tr><tr>
<td>
`default_value`
</td>
<td>
The value to use for out-of-vocabulary feature values.
Defaults to -1.
</td>
</tr><tr>
<td>
`hasher_spec`
</td>
<td>
A `HasherSpec` to specify the hash function to use for
assignation of out-of-vocabulary buckets.
</td>
</tr><tr>
<td>
`key_dtype`
</td>
<td>
The `key` data type.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for this op (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The lookup table based on the given dataset.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If
* `dataset` does not contain pairs
* The 2nd item in the `dataset` pairs has a dtype which is incompatible
  with `default_value`
* `num_oov_buckets` is negative
* `vocab_size` is not greater than zero
* The `key_dtype` is not integer or string
</td>
</tr>
</table>

