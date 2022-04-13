description: Represents real valued or numerical features.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.feature_column.numeric_column" />
<meta itemprop="path" content="Stable" />
</div>

# tf.feature_column.numeric_column

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/feature_column/feature_column_v2.py">View source</a>



Represents real valued or numerical features.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.feature_column.numeric_column`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.feature_column.numeric_column(
    key, shape=(1,), default_value=None, dtype=tf.dtypes.float32, normalizer_fn=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Example:



Assume we have data with two features `a` and `b`.

```
>>> data = {'a': [15, 9, 17, 19, 21, 18, 25, 30],
...    'b': [5.0, 6.4, 10.5, 13.6, 15.7, 19.9, 20.3 , 0.0]}
```

Let us represent the features `a` and `b` as numerical features.

```
>>> a = tf.feature_column.numeric_column('a')
>>> b = tf.feature_column.numeric_column('b')
```

Feature column describe a set of transformations to the inputs.

For example, to "bucketize" feature `a`, wrap the `a` column in a
<a href="../../tf/feature_column/bucketized_column.md"><code>feature_column.bucketized_column</code></a>.
Providing `5` bucket boundaries, the bucketized_column api
will bucket this feature in total of `6` buckets.

```
>>> a_buckets = tf.feature_column.bucketized_column(a,
...    boundaries=[10, 15, 20, 25, 30])
```

Create a `DenseFeatures` layer which will apply the transformations
described by the set of <a href="../../tf/feature_column.md"><code>tf.feature_column</code></a> objects:

```
>>> feature_layer = tf.keras.layers.DenseFeatures([a_buckets, b])
>>> print(feature_layer(data))
tf.Tensor(
[[ 0.   0.   1.   0.   0.   0.   5. ]
 [ 1.   0.   0.   0.   0.   0.   6.4]
 [ 0.   0.   1.   0.   0.   0.  10.5]
 [ 0.   0.   1.   0.   0.   0.  13.6]
 [ 0.   0.   0.   1.   0.   0.  15.7]
 [ 0.   0.   1.   0.   0.   0.  19.9]
 [ 0.   0.   0.   0.   1.   0.  20.3]
 [ 0.   0.   0.   0.   0.   1.   0. ]], shape=(8, 7), dtype=float32)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`key`
</td>
<td>
A unique string identifying the input feature. It is used as the
column name and the dictionary key for feature parsing configs, feature
`Tensor` objects, and feature columns.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
An iterable of integers specifies the shape of the `Tensor`. An
integer can be given which means a single dimension `Tensor` with given
width. The `Tensor` representing the column will have the shape of
[batch_size] + `shape`.
</td>
</tr><tr>
<td>
`default_value`
</td>
<td>
A single value compatible with `dtype` or an iterable of
values compatible with `dtype` which the column takes on during
`tf.Example` parsing if data is missing. A default value of `None` will
cause <a href="../../tf/io/parse_example.md"><code>tf.io.parse_example</code></a> to fail if an example does not contain this
column. If a single value is provided, the same value will be applied as
the default value for every item. If an iterable of values is provided,
the shape of the `default_value` should be equal to the given `shape`.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
defines the type of values. Default value is <a href="../../tf.md#float32"><code>tf.float32</code></a>. Must be a
non-quantized, real integer or floating point type.
</td>
</tr><tr>
<td>
`normalizer_fn`
</td>
<td>
If not `None`, a function that can be used to normalize the
value of the tensor after `default_value` is applied for parsing.
Normalizer function takes the input `Tensor` as its argument, and returns
the output `Tensor`. (e.g. lambda x: (x - 3.0) / 4.2). Please note that
even though the most common use case of this function is normalization, it
can be used for any kind of Tensorflow transformations.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `NumericColumn`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
if any dimension in shape is not an int
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
if any dimension in shape is not a positive integer
</td>
</tr><tr>
<td>
`TypeError`
</td>
<td>
if `default_value` is an iterable but not compatible with `shape`
</td>
</tr><tr>
<td>
`TypeError`
</td>
<td>
if `default_value` is not compatible with `dtype`.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
if `dtype` is not convertible to <a href="../../tf.md#float32"><code>tf.float32</code></a>.
</td>
</tr>
</table>

