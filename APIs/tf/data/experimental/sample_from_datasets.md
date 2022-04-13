description: Samples elements at random from the datasets in datasets. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.sample_from_datasets" />
<meta itemprop="path" content="Stable" />
</div>

# tf.data.experimental.sample_from_datasets

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/data/experimental/ops/interleave_ops.py">View source</a>



Samples elements at random from the datasets in `datasets`. (deprecated)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.sample_from_datasets(
    datasets, weights=None, seed=None, stop_on_empty_dataset=(False)
)
</code></pre>



<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use <a href="../../../tf/data/Dataset.md#sample_from_datasets"><code>tf.data.Dataset.sample_from_datasets(...)</code></a>.

Creates a dataset by interleaving elements of `datasets` with `weight[i]`
probability of picking an element from dataset `i`. Sampling is done without
replacement. For example, suppose we have 2 datasets:

```python
dataset1 = tf.data.Dataset.range(0, 3)
dataset2 = tf.data.Dataset.range(100, 103)
```

Suppose also that we sample from these 2 datasets with the following weights:

```python
sample_dataset = tf.data.Dataset.sample_from_datasets(
    [dataset1, dataset2], weights=[0.5, 0.5])
```

One possible outcome of elements in sample_dataset is:

```
print(list(sample_dataset.as_numpy_iterator()))
# [100, 0, 1, 101, 2, 102]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`datasets`
</td>
<td>
A non-empty list of <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> objects with compatible
structure.
</td>
</tr><tr>
<td>
`weights`
</td>
<td>
(Optional.) A list or Tensor of `len(datasets)` floating-point
values where `weights[i]` represents the probability to sample from
`datasets[i]`, or a <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> object where each element is such a
list. Defaults to a uniform distribution across `datasets`.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
(Optional.) A <a href="../../../tf.md#int64"><code>tf.int64</code></a> scalar <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>, representing the random
seed that will be used to create the distribution. See
<a href="../../../tf/random/set_seed.md"><code>tf.random.set_seed</code></a> for behavior.
</td>
</tr><tr>
<td>
`stop_on_empty_dataset`
</td>
<td>
If `True`, sampling stops if it encounters an empty
dataset. If `False`, it skips empty datasets. It is recommended to set it
to `True`. Otherwise, the distribution of samples starts off as the user
intends, but may change as input datasets become empty. This can be
difficult to detect since the dataset starts off looking correct. Default
to `False` for backward compatibility.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dataset that interleaves elements from `datasets` at random, according to
`weights` if provided, otherwise with uniform probability.
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
If the `datasets` or `weights` arguments have the wrong type.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
  - If `datasets` is empty, or
- If `weights` is specified and does not match the length of `datasets`.
</td>
</tr>
</table>

