description: Adds a Sum-of-Squares loss to the training procedure.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.losses.mean_squared_error" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.losses.mean_squared_error

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/losses/losses_impl.py">View source</a>



Adds a Sum-of-Squares loss to the training procedure.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.losses.mean_squared_error(
    labels, predictions, weights=1.0, scope=None,
    loss_collection=tf.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

<a href="../../../../tf/compat/v1/losses/mean_squared_error.md"><code>tf.compat.v1.losses.mean_squared_error</code></a> is mostly compatible with eager
execution and <a href="../../../../tf/function.md"><code>tf.function</code></a>. But, the `loss_collection` argument is
ignored when executing eagerly and no loss will be written to the loss
collections. You will need to either hold on to the return value manually
or rely on <a href="../../../../tf/keras/Model.md"><code>tf.keras.Model</code></a> loss tracking.


To switch to native TF2 style, instantiate the
 <a href="../../../../tf/keras/losses/MeanSquaredError.md"><code>tf.keras.losses.MeanSquaredError</code></a> class and call the object instead.


#### Structural Mapping to Native TF2

Before:

```python
loss = tf.compat.v1.losses.mean_squared_error(
  labels=labels,
  predictions=predictions,
  weights=weights,
  reduction=reduction)
```

After:

```python
loss_fn = tf.keras.losses.MeanSquaredError(
  reduction=reduction)
loss = loss_fn(
  y_true=labels,
  y_pred=predictions,
  sample_weight=weights)
```

#### How to Map Arguments

| TF1 Arg Name          | TF2 Arg Name     | Note                       |
| :-------------------- | :--------------- | :------------------------- |
| `labels`              | `y_true`         | In `__call__()` method     |
| `predictions`         | `y_pred`         | In `__call__()` method     |
| `weights`             | `sample_weight`  | In `__call__()` method.    |
: : : The shape requirements for `sample_weight` is different from      :
: : : `weights`. Please check the [argument definition][api_docs] for   :
: : : details.                                                          :
| `scope`               | Not supported    | -                          |
| `loss_collection`     | Not supported    | Losses should be tracked   |
: : : explicitly or with Keras APIs, for example, [add_loss][add_loss], :
: : : instead of via collections                                        :
| `reduction`           | `reduction`      | In constructor. Value of   |
: : : <a href="../../../../tf/compat/v1/losses/Reduction.md#SUM_OVER_BATCH_SIZE"><code>tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE</code></a>,              :
: : : <a href="../../../../tf/compat/v1/losses/Reduction.md#SUM"><code>tf.compat.v1.losses.Reduction.SUM</code></a>,                              :
: : : <a href="../../../../tf/compat/v1/losses/Reduction.md#NONE"><code>tf.compat.v1.losses.Reduction.NONE</code></a> in                           :
: : : <a href="../../../../tf/compat/v1/losses/softmax_cross_entropy.md"><code>tf.compat.v1.losses.softmax_cross_entropy</code></a> correspond to         :
: : : <a href="../../../../tf/keras/losses/Reduction.md#SUM_OVER_BATCH_SIZE"><code>tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE</code></a>,                  :
: : : <a href="../../../../tf/keras/losses/Reduction.md#SUM"><code>tf.keras.losses.Reduction.SUM</code></a>,                                  :
: : : <a href="../../../../tf/keras/losses/Reduction.md#NONE"><code>tf.keras.losses.Reduction.NONE</code></a>, respectively. If you            :
: : : used other value for `reduction`, including the default value     :
: : :  <a href="../../../../tf/compat/v1/losses/Reduction.md#SUM_BY_NONZERO_WEIGHTS"><code>tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS</code></a>, there is :
: : : no directly corresponding value. Please modify the loss           :
: : : implementation manually.                                          :

[add_loss]:https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_loss
[api_docs]:https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError#__call__


#### Before & After Usage Example

Before:

```
>>> y_true = [1, 2, 3]
>>> y_pred = [1, 3, 5]
>>> weights = [0, 1, 0.25]
>>> # samples with zero-weight are excluded from calculation when `reduction`
>>> # argument is set to default value `Reduction.SUM_BY_NONZERO_WEIGHTS`
>>> tf.compat.v1.losses.mean_squared_error(
...    labels=y_true,
...    predictions=y_pred,
...    weights=weights).numpy()
1.0
```

```
>>> tf.compat.v1.losses.mean_squared_error(
...    labels=y_true,
...    predictions=y_pred,
...    weights=weights,
...    reduction=tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE).numpy()
0.66667
```

After:

```
>>> y_true = [[1.0], [2.0], [3.0]]
>>> y_pred = [[1.0], [3.0], [5.0]]
>>> weights = [1, 1, 0.25]
>>> mse = tf.keras.losses.MeanSquaredError(
...    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
>>> mse(y_true=y_true, y_pred=y_pred, sample_weight=weights).numpy()
0.66667
```



 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

`weights` acts as a coefficient for the loss. If a scalar is provided, then
the loss is simply scaled by the given value. If `weights` is a tensor of size
`[batch_size]`, then the total loss for each sample of the batch is rescaled
by the corresponding element in the `weights` vector. If the shape of
`weights` matches the shape of `predictions`, then the loss of each
measurable element of `predictions` is scaled by the corresponding value of
`weights`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`labels`
</td>
<td>
The ground truth output tensor, same dimensions as 'predictions'.
</td>
</tr><tr>
<td>
`predictions`
</td>
<td>
The predicted outputs.
</td>
</tr><tr>
<td>
`weights`
</td>
<td>
Optional `Tensor` whose rank is either 0, or the same rank as
`labels`, and must be broadcastable to `labels` (i.e., all dimensions must
be either `1`, or the same as the corresponding `losses` dimension).
</td>
</tr><tr>
<td>
`scope`
</td>
<td>
The scope for the operations performed in computing the loss.
</td>
</tr><tr>
<td>
`loss_collection`
</td>
<td>
collection to which the loss will be added.
</td>
</tr><tr>
<td>
`reduction`
</td>
<td>
Type of reduction to apply to loss.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
shape as `labels`; otherwise, it is scalar.
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
If the shape of `predictions` doesn't match that of `labels` or
if the shape of `weights` is invalid.  Also if `labels` or `predictions`
is None.
</td>
</tr>
</table>


