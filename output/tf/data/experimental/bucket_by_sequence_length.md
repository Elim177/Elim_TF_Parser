description: A transformation that buckets elements in a Dataset by length. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.bucket_by_sequence_length" />
<meta itemprop="path" content="Stable" />
</div>

# tf.data.experimental.bucket_by_sequence_length

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/data/experimental/ops/grouping.py">View source</a>



A transformation that buckets elements in a `Dataset` by length. (deprecated)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.data.experimental.bucket_by_sequence_length`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.bucket_by_sequence_length(
    element_length_func, bucket_boundaries, bucket_batch_sizes, padded_shapes=None,
    padding_values=None, pad_to_bucket_boundary=(False), no_padding=(False),
    drop_remainder=(False)
)
</code></pre>



<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use <a href="../../../tf/data/Dataset.md#bucket_by_sequence_length"><code>tf.data.Dataset.bucket_by_sequence_length(...)</code></a>.

Elements of the `Dataset` are grouped together by length and then are padded
and batched.

This is useful for sequence tasks in which the elements have variable length.
Grouping together elements that have similar lengths reduces the total
fraction of padding in a batch which increases training step efficiency.

Below is an example to bucketize the input data to the 3 buckets
"[0, 3), [3, 5), [5, inf)" based on sequence length, with batch size 2.

```
>>> elements = [
...   [0], [1, 2, 3, 4], [5, 6, 7],
...   [7, 8, 9, 10, 11], [13, 14, 15, 16, 19, 20], [21, 22]]
```

```
>>> dataset = tf.data.Dataset.from_generator(
...     lambda: elements, tf.int64, output_shapes=[None])
```

```
>>> dataset = dataset.apply(
...     tf.data.experimental.bucket_by_sequence_length(
...         element_length_func=lambda elem: tf.shape(elem)[0],
...         bucket_boundaries=[3, 5],
...         bucket_batch_sizes=[2, 2, 2]))
```

```
>>> for elem in dataset.as_numpy_iterator():
...   print(elem)
[[1 2 3 4]
 [5 6 7 0]]
[[ 7  8  9 10 11  0]
 [13 14 15 16 19 20]]
[[ 0  0]
 [21 22]]
```

There is also a possibility to pad the dataset till the bucket boundary.
You can also provide which value to be used while padding the data.
Below example uses `-1` as padding and it also shows the input data
being bucketizied to two buckets "[0,3], [4,6]".

```
>>> elements = [
...   [0], [1, 2, 3, 4], [5, 6, 7],
...   [7, 8, 9, 10, 11], [13, 14, 15, 16, 19, 20], [21, 22]]
```

```
>>> dataset = tf.data.Dataset.from_generator(
...   lambda: elements, tf.int32, output_shapes=[None])
```

```
>>> dataset = dataset.apply(
...     tf.data.experimental.bucket_by_sequence_length(
...         element_length_func=lambda elem: tf.shape(elem)[0],
...         bucket_boundaries=[4, 7],
...         bucket_batch_sizes=[2, 2, 2],
...         pad_to_bucket_boundary=True,
...         padding_values=-1))
```

```
>>> for elem in dataset.as_numpy_iterator():
...   print(elem)
[[ 0 -1 -1]
 [ 5  6  7]]
[[ 1  2  3  4 -1 -1]
 [ 7  8  9 10 11 -1]]
[[21 22 -1]]
[[13 14 15 16 19 20]]
```

When using `pad_to_bucket_boundary` option, it can be seen that it is
not always possible to maintain the bucket batch size.
You can drop the batches that do not maintain the bucket batch size by
using the option `drop_remainder`. Using the same input data as in the
above example you get the following result.

```
>>> elements = [
...   [0], [1, 2, 3, 4], [5, 6, 7],
...   [7, 8, 9, 10, 11], [13, 14, 15, 16, 19, 20], [21, 22]]
```

```
>>> dataset = tf.data.Dataset.from_generator(
...   lambda: elements, tf.int32, output_shapes=[None])
```

```
>>> dataset = dataset.apply(
...     tf.data.experimental.bucket_by_sequence_length(
...         element_length_func=lambda elem: tf.shape(elem)[0],
...         bucket_boundaries=[4, 7],
...         bucket_batch_sizes=[2, 2, 2],
...         pad_to_bucket_boundary=True,
...         padding_values=-1,
...         drop_remainder=True))
```

```
>>> for elem in dataset.as_numpy_iterator():
...   print(elem)
[[ 0 -1 -1]
 [ 5  6  7]]
[[ 1  2  3  4 -1 -1]
 [ 7  8  9 10 11 -1]]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`element_length_func`
</td>
<td>
function from element in `Dataset` to <a href="../../../tf.md#int32"><code>tf.int32</code></a>,
determines the length of the element, which will determine the bucket it
goes into.
</td>
</tr><tr>
<td>
`bucket_boundaries`
</td>
<td>
`list<int>`, upper length boundaries of the buckets.
</td>
</tr><tr>
<td>
`bucket_batch_sizes`
</td>
<td>
`list<int>`, batch size per bucket. Length should be
`len(bucket_boundaries) + 1`.
</td>
</tr><tr>
<td>
`padded_shapes`
</td>
<td>
Nested structure of <a href="../../../tf/TensorShape.md"><code>tf.TensorShape</code></a> to pass to
<a href="../../../tf/data/Dataset.md#padded_batch"><code>tf.data.Dataset.padded_batch</code></a>. If not provided, will use
`dataset.output_shapes`, which will result in variable length dimensions
being padded out to the maximum length in each batch.
</td>
</tr><tr>
<td>
`padding_values`
</td>
<td>
Values to pad with, passed to
<a href="../../../tf/data/Dataset.md#padded_batch"><code>tf.data.Dataset.padded_batch</code></a>. Defaults to padding with 0.
</td>
</tr><tr>
<td>
`pad_to_bucket_boundary`
</td>
<td>
bool, if `False`, will pad dimensions with unknown
size to maximum length in batch. If `True`, will pad dimensions with
unknown size to bucket boundary minus 1 (i.e., the maximum length in each
bucket), and caller must ensure that the source `Dataset` does not contain
any elements with length longer than `max(bucket_boundaries)`.
</td>
</tr><tr>
<td>
`no_padding`
</td>
<td>
`bool`, indicates whether to pad the batch features (features
need to be either of type <a href="../../../tf/sparse/SparseTensor.md"><code>tf.sparse.SparseTensor</code></a> or of same shape).
</td>
</tr><tr>
<td>
`drop_remainder`
</td>
<td>
(Optional.) A <a href="../../../tf.md#bool"><code>tf.bool</code></a> scalar <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>, representing
whether the last batch should be dropped in the case it has fewer than
`batch_size` elements; the default behavior is not to drop the smaller
batch.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Dataset` transformation function, which can be passed to
<a href="../../../tf/data/Dataset.md#apply"><code>tf.data.Dataset.apply</code></a>.
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
if `len(bucket_batch_sizes) != len(bucket_boundaries) + 1`.
</td>
</tr>
</table>

