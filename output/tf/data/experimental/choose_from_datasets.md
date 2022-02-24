description: Creates a dataset that deterministically chooses elements from datasets. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.choose_from_datasets" />
<meta itemprop="path" content="Stable" />
</div>

# tf.data.experimental.choose_from_datasets

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/data/experimental/ops/interleave_ops.py">View source</a>



Creates a dataset that deterministically chooses elements from `datasets`. (deprecated)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.choose_from_datasets(
    datasets, choice_dataset, stop_on_empty_dataset=(False)
)
</code></pre>



<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use <a href="../../../tf/data/Dataset.md#choose_from_datasets"><code>tf.data.Dataset.choose_from_datasets(...)</code></a> instead. Note that, unlike the experimental endpoint, the non-experimental endpoint sets `stop_on_empty_dataset=True` by default. You should set this argument explicitly in case you would like to match the behavior of the experimental endpoint.

For example, given the following datasets:

```python
datasets = [tf.data.Dataset.from_tensors("foo").repeat(),
            tf.data.Dataset.from_tensors("bar").repeat(),
            tf.data.Dataset.from_tensors("baz").repeat()]

# Define a dataset containing `[0, 1, 2, 0, 1, 2, 0, 1, 2]`.
choice_dataset = tf.data.Dataset.range(3).repeat(3)

result = tf.data.experimental.choose_from_datasets(datasets, choice_dataset)
```

The elements of `result` will be:

```
"foo", "bar", "baz", "foo", "bar", "baz", "foo", "bar", "baz"
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
`choice_dataset`
</td>
<td>
A <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> of scalar <a href="../../../tf.md#int64"><code>tf.int64</code></a> tensors between `0`
and `len(datasets) - 1`.
</td>
</tr><tr>
<td>
`stop_on_empty_dataset`
</td>
<td>
If `True`, selection stops if it encounters an empty
dataset. If `False`, it skips empty datasets. It is recommended to set it
to `True`. Otherwise, the selected elements start off as the user intends,
but may change as input datasets become empty. This can be difficult to
detect since the dataset starts off looking correct. Default to `False`
for backward compatibility.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dataset that interleaves elements from `datasets` according to the values
of `choice_dataset`.
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
If `datasets` or `choice_dataset` has the wrong type.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If `datasets` is empty.
</td>
</tr>
</table>

