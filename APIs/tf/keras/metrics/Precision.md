description: Computes the precision of the predictions with respect to the labels.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.metrics.Precision" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="merge_state"/>
<meta itemprop="property" content="reset_state"/>
<meta itemprop="property" content="result"/>
<meta itemprop="property" content="update_state"/>
</div>

# tf.keras.metrics.Precision

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/metrics.py#L1291-L1431">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes the precision of the predictions with respect to the labels.

Inherits From: [`Metric`](../../../tf/keras/metrics/Metric.md), [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.metrics.Precision`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.metrics.Precision`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.metrics.Precision(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The metric creates two local variables, `true_positives` and `false_positives`
that are used to compute the precision. This value is ultimately returned as
`precision`, an idempotent operation that simply divides `true_positives`
by the sum of `true_positives` and `false_positives`.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

If `top_k` is set, we'll calculate precision as how often on average a class
among the top-k classes with the highest predicted values of a batch entry is
correct and can be found in the label for that entry.

If `class_id` is specified, we calculate precision by considering only the
entries in the batch for which `class_id` is above the threshold and/or in the
top-k highest predictions, and computing the fraction of them for which
`class_id` is indeed a correct label.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`thresholds`
</td>
<td>
(Optional) A float value or a python list/tuple of float
threshold values in [0, 1]. A threshold is compared with prediction
values to determine the truth value of predictions (i.e., above the
threshold is `true`, below is `false`). One metric value is generated
for each threshold value. If neither thresholds nor top_k are set, the
default is to calculate precision with `thresholds=0.5`.
</td>
</tr><tr>
<td>
`top_k`
</td>
<td>
(Optional) Unset by default. An int value specifying the top-k
predictions to consider when calculating precision.
</td>
</tr><tr>
<td>
`class_id`
</td>
<td>
(Optional) Integer class ID for which we want binary metrics.
This must be in the half-open interval `[0, num_classes)`, where
`num_classes` is the last dimension of predictions.
</td>
</tr><tr>
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
</tr>
</table>



#### Standalone usage:



```
>>> m = tf.keras.metrics.Precision()
>>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
>>> m.result().numpy()
0.6666667
```

```
>>> m.reset_state()
>>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
>>> m.result().numpy()
1.0
```

```
>>> # With top_k=2, it will calculate precision over y_true[:2] and y_pred[:2]
>>> m = tf.keras.metrics.Precision(top_k=2)
>>> m.update_state([0, 0, 1, 1], [1, 1, 1, 1])
>>> m.result().numpy()
0.0
```

```
>>> # With top_k=4, it will calculate precision over y_true[:4] and y_pred[:4]
>>> m = tf.keras.metrics.Precision(top_k=4)
>>> m.update_state([0, 0, 1, 1], [1, 1, 1, 1])
>>> m.result().numpy()
0.5
```

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='mse',
              metrics=[tf.keras.metrics.Precision()])
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

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/metrics.py#L1418-L1422">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_state()
</code></pre>

Resets all of the metric state variables.

This function is called between epochs/steps,
when a metric is evaluated during training.

<h3 id="result"><code>result</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/metrics.py#L1412-L1416">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>result()
</code></pre>

Computes and returns the metric value tensor.

Result computation is an idempotent operation that simply calculates the
metric value using the state variables.

<h3 id="update_state"><code>update_state</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/metrics.py#L1385-L1410">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_state(
    y_true, y_pred, sample_weight=None
)
</code></pre>

Accumulates true positive and false positive statistics.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`y_true`
</td>
<td>
The ground truth values, with the same dimensions as `y_pred`.
Will be cast to `bool`.
</td>
</tr><tr>
<td>
`y_pred`
</td>
<td>
The predicted values. Each element must be in the range `[0, 1]`.
</td>
</tr><tr>
<td>
`sample_weight`
</td>
<td>
Optional weighting of each example. Defaults to 1. Can be a
`Tensor` whose rank is either 0, or the same rank as `y_true`, and must
be broadcastable to `y_true`.
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





