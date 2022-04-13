description: Creates a Dataset that counts from start in steps of size step.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.Counter" />
<meta itemprop="path" content="Stable" />
</div>

# tf.data.experimental.Counter

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/data/experimental/ops/counter.py">View source</a>



Creates a `Dataset` that counts from `start` in steps of size `step`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.Counter(
    start=0, step=1, dtype=tf.dtypes.int64
)
</code></pre>



<!-- Placeholder for "Used in" -->

Unlike <a href="../../../tf/data/Dataset.md#range"><code>tf.data.Dataset.range</code></a> which will stop at some ending number,
`Counter` will produce elements indefinitely.

```
>>> dataset = tf.data.experimental.Counter().take(5)
>>> list(dataset.as_numpy_iterator())
[0, 1, 2, 3, 4]
>>> dataset.element_spec
TensorSpec(shape=(), dtype=tf.int64, name=None)
>>> dataset = tf.data.experimental.Counter(dtype=tf.int32)
>>> dataset.element_spec
TensorSpec(shape=(), dtype=tf.int32, name=None)
>>> dataset = tf.data.experimental.Counter(start=2).take(5)
>>> list(dataset.as_numpy_iterator())
[2, 3, 4, 5, 6]
>>> dataset = tf.data.experimental.Counter(start=2, step=5).take(5)
>>> list(dataset.as_numpy_iterator())
[2, 7, 12, 17, 22]
>>> dataset = tf.data.experimental.Counter(start=10, step=-1).take(5)
>>> list(dataset.as_numpy_iterator())
[10, 9, 8, 7, 6]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`start`
</td>
<td>
(Optional.) The starting value for the counter. Defaults to 0.
</td>
</tr><tr>
<td>
`step`
</td>
<td>
(Optional.) The step size for the counter. Defaults to 1.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
(Optional.) The data type for counter elements. Defaults to
<a href="../../../tf.md#int64"><code>tf.int64</code></a>.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Dataset` of scalar `dtype` elements.
</td>
</tr>

</table>

