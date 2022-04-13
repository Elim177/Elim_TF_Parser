description: A CategoricalColumn with a vocabulary file.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.feature_column.categorical_column_with_vocabulary_file" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.feature_column.categorical_column_with_vocabulary_file

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/feature_column/feature_column_v2.py">View source</a>



A `CategoricalColumn` with a vocabulary file.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.feature_column.categorical_column_with_vocabulary_file(
    key, vocabulary_file, vocabulary_size=None, num_oov_buckets=0,
    default_value=None, dtype=tf.dtypes.string
)
</code></pre>



<!-- Placeholder for "Used in" -->

Use this when your inputs are in string or integer format, and you have a
vocabulary file that maps each value to an integer ID. By default,
out-of-vocabulary values are ignored. Use either (but not both) of
`num_oov_buckets` and `default_value` to specify how to include
out-of-vocabulary values.

For input dictionary `features`, `features[key]` is either `Tensor` or
`SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
and `''` for string, which will be dropped by this feature column.

Example with `num_oov_buckets`:
File '/us/states.txt' contains 50 lines, each with a 2-character U.S. state
abbreviation. All inputs with values in that file are assigned an ID 0-49,
corresponding to its line number. All other values are hashed and assigned an
ID 50-54.

```python
import tensorflow as tf
states = tf.feature_column.categorical_column_with_vocabulary_file(
  key='states', vocabulary_file='states.txt', vocabulary_size=5,
  num_oov_buckets=1)
columns = [states]
features = {'states':tf.constant([['california', 'georgia', 'michigan',
'texas', 'new york'], ['new york', 'georgia', 'california', 'michigan',
'texas']])}
linear_prediction = tf.compat.v1.feature_column.linear_model(features,
columns)
```

Example with `default_value`:
File '/us/states.txt' contains 51 lines - the first line is 'XX', and the
other 50 each have a 2-character U.S. state abbreviation. Both a literal 'XX'
in input, and other values missing from the file, will be assigned ID 0. All
others are assigned the corresponding line number 1-50.

```python
import tensorflow as tf
states = tf.feature_column.categorical_column_with_vocabulary_file(
  key='states', vocabulary_file='states.txt', vocabulary_size=6,
  default_value=0)
columns = [states]
features = {'states':tf.constant([['california', 'georgia', 'michigan',
'texas', 'new york'], ['new york', 'georgia', 'california', 'michigan',
'texas']])}
linear_prediction = tf.compat.v1.feature_column.linear_model(features,
columns)
```

And to make an embedding with either:

```python
import tensorflow as tf
states = tf.feature_column.categorical_column_with_vocabulary_file(
  key='states', vocabulary_file='states.txt', vocabulary_size=5,
  num_oov_buckets=1)
columns = [tf.feature_column.embedding_column(states, 3)]
features = {'states':tf.constant([['california', 'georgia', 'michigan',
'texas', 'new york'], ['new york', 'georgia', 'california', 'michigan',
'texas']])}
input_layer = tf.keras.layers.DenseFeatures(columns)
dense_tensor = input_layer(features)
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
`vocabulary_file`
</td>
<td>
The vocabulary file name.
</td>
</tr><tr>
<td>
`vocabulary_size`
</td>
<td>
Number of the elements in the vocabulary. This must be no
greater than length of `vocabulary_file`, if less than length, later
values are ignored. If None, it is set to the length of `vocabulary_file`.
</td>
</tr><tr>
<td>
`num_oov_buckets`
</td>
<td>
Non-negative integer, the number of out-of-vocabulary
buckets. All out-of-vocabulary inputs will be assigned IDs in the range
`[vocabulary_size, vocabulary_size+num_oov_buckets)` based on a hash of
the input value. A positive `num_oov_buckets` can not be specified with
`default_value`.
</td>
</tr><tr>
<td>
`default_value`
</td>
<td>
The integer ID value to return for out-of-vocabulary feature
values, defaults to `-1`. This can not be specified with a positive
`num_oov_buckets`.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The type of features. Only string and integer types are supported.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `CategoricalColumn` with a vocabulary file.
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
`vocabulary_file` is missing or cannot be opened.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
`vocabulary_size` is missing or < 1.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
`num_oov_buckets` is a negative integer.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
`num_oov_buckets` and `default_value` are both specified.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
`dtype` is neither string nor integer.
</td>
</tr>
</table>

