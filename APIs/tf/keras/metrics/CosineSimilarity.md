description: Computes the cosine similarity between the labels and predictions.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.metrics.CosineSimilarity" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="merge_state"/>
<meta itemprop="property" content="reset_state"/>
<meta itemprop="property" content="result"/>
<meta itemprop="property" content="update_state"/>
</div>

# tf.keras.metrics.CosineSimilarity

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/metrics.py#L2532-L2579">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes the cosine similarity between the labels and predictions.

Inherits From: [`MeanMetricWrapper`](../../../tf/keras/metrics/MeanMetricWrapper.md), [`Mean`](../../../tf/keras/metrics/Mean.md), [`Metric`](../../../tf/keras/metrics/Metric.md), [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.metrics.CosineSimilarity`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.metrics.CosineSimilarity`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.metrics.CosineSimilarity(
    name=&#x27;cosine_similarity&#x27;, dtype=None, axis=-1
)
</code></pre>



<!-- Placeholder for "Used in" -->

`cosine similarity = (a . b) / ||a|| ||b||`

See: [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity).

This metric keeps the average cosine similarity between `predictions` and
`labels` over a stream of data.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`
</td>
<td>
(Optional) string name of the metric instance.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
(Optional) data type of the metric result.
</td>
</tr><tr>
<td>
`axis`
</td>
<td>
(Optional) Defaults to -1. The dimension along which the cosine
similarity is computed.
</td>
</tr>
</table>



#### Standalone usage:



```
>>> # l2_norm(y_true) = [[0., 1.], [1./1.414, 1./1.414]]
>>> # l2_norm(y_pred) = [[1., 0.], [1./1.414, 1./1.414]]
>>> # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]]
>>> # result = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1))
>>> #        = ((0. + 0.) +  (0.5 + 0.5)) / 2
>>> m = tf.keras.metrics.CosineSimilarity(axis=1)
>>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]])
>>> m.result().numpy()
0.49999997
```

```
>>> m.reset_state()
>>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]],
...                sample_weight=[0.3, 0.7])
>>> m.result().numpy()
0.6999999
```

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[tf.keras.metrics.CosineSimilarity(axis=1)])
```

## Methods

<h3 id="merge_state"><code>merge_state</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/metrics.py#L288-L322">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>merge_state(
    metrics
)
</code></pre>

Merges the state from one or more metrics.

This method can be used by distributed systems to merge the state computed
by different metric instances. Typically the state will be stored in the
form of the metric's weights. For example, a tf.keras.metrics.Mean metric
contains a list of two weight values: a total and a count. If there were two
instances of a tf.keras.metrics.Accuracy that each independently aggregated
partial state for an overall accuracy calculation, these two metric's states
could be combined as follows:

```
>>> m1 = tf.keras.metrics.Accuracy()
>>> _ = m1.update_state([[1], [2]], [[0], [2]])
```

```
>>> m2 = tf.keras.metrics.Accuracy()
>>> _ = m2.update_state([[3], [4]], [[3], [4]])
```

```
>>> m2.merge_state([m1])
>>> m2.result().numpy()
0.75
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`metrics`
</td>
<td>
an iterable of metrics. The metrics must have compatible state.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If the provided iterable does not contain metrics matching the
metric's required specifications.
</td>
</tr>
</table>



<h3 id="reset_state"><code>reset_state</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/metrics.py#L251-L266">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_state()
</code></pre>

Resets all of the metric state variables.

This function is called between epochs/steps,
when a metric is evaluated during training.

<h3 id="result"><code>result</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/metrics.py#L489-L500">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>result()
</code></pre>

Computes and returns the metric value tensor.

Result computation is an idempotent operation that simply calculates the
metric value using the state variables.

<h3 id="update_state"><code>update_state</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/metrics.py#L695-L727">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_state(
    y_true, y_pred, sample_weight=None
)
</code></pre>

Accumulates metric statistics.

For sparse categorical metrics, the shapes of `y_true` and `y_pred` are
different.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`y_true`
</td>
<td>
Ground truth label values. shape = `[batch_size, d0, .. dN-1]` or
shape = `[batch_size, d0, .. dN-1, 1]`.
</td>
</tr><tr>
<td>
`y_pred`
</td>
<td>
The predicted probability values. shape = `[batch_size, d0, .. dN]`.
</td>
</tr><tr>
<td>
`sample_weight`
</td>
<td>
Optional `sample_weight` acts as a
coefficient for the metric. If a scalar is provided, then the metric is
simply scaled by the given value. If `sample_weight` is a tensor of size
`[batch_size]`, then the metric for each sample of the batch is rescaled
by the corresponding element in the `sample_weight` vector. If the shape
of `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted
to this shape), then each metric element of `y_pred` is scaled by the
corresponding value of `sample_weight`. (Note on `dN-1`: all metric
functions reduce by 1 dimension, usually the last axis (-1)).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Update op.
</td>
</tr>

</table>




