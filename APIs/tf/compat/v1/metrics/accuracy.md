description: Calculates how often predictions matches labels.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.metrics.accuracy" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.metrics.accuracy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/metrics_impl.py">View source</a>



Calculates how often `predictions` matches `labels`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.metrics.accuracy(
    labels, predictions, weights=None, metrics_collections=None,
    updates_collections=None, name=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

<a href="../../../../tf/compat/v1/metrics/accuracy.md"><code>tf.compat.v1.metrics.accuracy</code></a> is not compatible with eager
execution or <a href="../../../../tf/function.md"><code>tf.function</code></a>.
Please use <a href="../../../../tf/keras/metrics/Accuracy.md"><code>tf.keras.metrics.Accuracy</code></a> instead for TF2 migration. After
instantiating a <a href="../../../../tf/keras/metrics/Accuracy.md"><code>tf.keras.metrics.Accuracy</code></a> object, you can first call the
`update_state()` method to record the prediction/labels, and then call the
`result()` method to get the accuracy eagerly. You can also attach it to a
Keras model when calling the `compile` method. Please refer to [this
guide](https://www.tensorflow.org/guide/migrate#new-style_metrics_and_losses)
for more details.

#### Structural Mapping to Native TF2

Before:

```python
accuracy, update_op = tf.compat.v1.metrics.accuracy(
  labels=labels,
  predictions=predictions,
  weights=weights,
  metrics_collections=metrics_collections,
  update_collections=update_collections,
  name=name)
```

After:

```python
 m = tf.keras.metrics.Accuracy(
   name=name,
   dtype=None)

 m.update_state(
 y_true=labels,
 y_pred=predictions,
 sample_weight=weights)

 accuracy = m.result()
```

#### How to Map Arguments

| TF1 Arg Name          | TF2 Arg Name    | Note                       |
| :-------------------- | :-------------- | :------------------------- |
| `label`               | `y_true`        | In `update_state()` method |
| `predictions`         | `y_true`        | In `update_state()` method |
| `weights`             | `sample_weight` | In `update_state()` method |
| `metrics_collections` | Not supported   | Metrics should be tracked  |
:                       :                 : explicitly or with Keras   :
:                       :                 : APIs, for example,         :
:                       :                 : [add_metric][add_metric],  :
:                       :                 : instead of via collections :
| `updates_collections` | Not supported   | -                          |
| `name`                | `name`          | In constructor             |

[add_metric]:https//www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_metric


#### Before & After Usage Example

Before:

```
>>> g = tf.Graph()
>>> with g.as_default():
...   logits = [1, 2, 3]
...   labels = [0, 2, 3]
...   acc, acc_op = tf.compat.v1.metrics.accuracy(logits, labels)
...   global_init = tf.compat.v1.global_variables_initializer()
...   local_init = tf.compat.v1.local_variables_initializer()
>>> sess = tf.compat.v1.Session(graph=g)
>>> sess.run([global_init, local_init])
>>> print(sess.run([acc, acc_op]))
[0.0, 0.66667]
```


After:

```
>>> m = tf.keras.metrics.Accuracy()
>>> m.update_state([1, 2, 3], [0, 2, 3])
>>> m.result().numpy()
0.66667
```

```python
# Used within Keras model
model.compile(optimizer='sgd',
              loss='mse',
              metrics=[tf.keras.metrics.Accuracy()])
```



 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

The `accuracy` function creates two local variables, `total` and
`count` that are used to compute the frequency with which `predictions`
matches `labels`. This frequency is ultimately returned as `accuracy`: an
idempotent operation that simply divides `total` by `count`.

For estimation of the metric over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the `accuracy`.
Internally, an `is_correct` operation computes a `Tensor` with elements 1.0
where the corresponding elements of `predictions` and `labels` match and 0.0
otherwise. Then `update_op` increments `total` with the reduced sum of the
product of `weights` and `is_correct`, and it increments `count` with the
reduced sum of `weights`.

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`labels`
</td>
<td>
The ground truth values, a `Tensor` whose shape matches
`predictions`.
</td>
</tr><tr>
<td>
`predictions`
</td>
<td>
The predicted values, a `Tensor` of any shape.
</td>
</tr><tr>
<td>
`weights`
</td>
<td>
Optional `Tensor` whose rank is either 0, or the same rank as
`labels`, and must be broadcastable to `labels` (i.e., all dimensions must
be either `1`, or the same as the corresponding `labels` dimension).
</td>
</tr><tr>
<td>
`metrics_collections`
</td>
<td>
An optional list of collections that `accuracy` should
be added to.
</td>
</tr><tr>
<td>
`updates_collections`
</td>
<td>
An optional list of collections that `update_op` should
be added to.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
An optional variable_scope name.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`accuracy`
</td>
<td>
A `Tensor` representing the accuracy, the value of `total` divided
by `count`.
</td>
</tr><tr>
<td>
`update_op`
</td>
<td>
An operation that increments the `total` and `count` variables
appropriately and whose value matches `accuracy`.
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
If `predictions` and `labels` have mismatched shapes, or if
`weights` is not `None` and its shape doesn't match `predictions`, or if
either `metrics_collections` or `updates_collections` are not a list or
tuple.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
If eager execution is enabled.
</td>
</tr>
</table>


