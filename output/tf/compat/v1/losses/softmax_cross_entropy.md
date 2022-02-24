description: Creates a cross-entropy loss using tf.nn.softmax_cross_entropy_with_logits_v2.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.losses.softmax_cross_entropy" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.losses.softmax_cross_entropy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/losses/losses_impl.py">View source</a>



Creates a cross-entropy loss using tf.nn.softmax_cross_entropy_with_logits_v2.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.losses.softmax_cross_entropy(
    onehot_labels, logits, weights=1.0, label_smoothing=0, scope=None,
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

<a href="../../../../tf/compat/v1/losses/softmax_cross_entropy.md"><code>tf.compat.v1.losses.softmax_cross_entropy</code></a> is mostly compatible with eager
execution and <a href="../../../../tf/function.md"><code>tf.function</code></a>. But, the `loss_collection` argument is
ignored when executing eagerly and no loss will be written to the loss
collections. You will need to either hold on to the return value manually
or rely on <a href="../../../../tf/keras/Model.md"><code>tf.keras.Model</code></a> loss tracking.


To switch to native TF2 style, instantiate the
 <a href="../../../../tf/keras/losses/CategoricalCrossentropy.md"><code>tf.keras.losses.CategoricalCrossentropy</code></a> class with `from_logits` set
as `True` and call the object instead.


#### Structural Mapping to Native TF2

Before:

```python
loss = tf.compat.v1.losses.softmax_cross_entropy(
  onehot_labels=onehot_labels,
  logits=logits,
  weights=weights,
  label_smoothing=smoothing)
```

After:

```python
loss_fn = tf.keras.losses.CategoricalCrossentropy(
  from_logits=True,
  label_smoothing=smoothing)
loss = loss_fn(
  y_true=onehot_labels,
  y_pred=logits,
  sample_weight=weights)
```

#### How to Map Arguments

| TF1 Arg Name          | TF2 Arg Name     | Note                       |
| :-------------------- | :--------------- | :------------------------- |
|  -                    | `from_logits`    | Set `from_logits` as True  |
:                       :                  : to have identical behavior :
| `onehot_labels`       | `y_true`         | In `__call__()` method     |
| `logits`              | `y_pred`         | In `__call__()` method     |
| `weights`             | `sample_weight`  | In `__call__()` method     |
| `label_smoothing`     | `label_smoothing`| In constructor             |
| `scope`               | Not supported    | -                          |
| `loss_collection`     | Not supported    | Losses should be tracked   |
:                       :                  : explicitly or with Keras   :
:                       :                  : APIs, for example,         :
:                       :                  : [add_loss][add_loss],      :
:                       :                  : instead of via collections :
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


#### Before & After Usage Example

Before:

```
>>> y_true = [[0, 1, 0], [0, 0, 1]]
>>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
>>> weights = [0.3, 0.7]
>>> smoothing = 0.2
>>> tf.compat.v1.losses.softmax_cross_entropy(y_true, y_pred, weights=weights,
...   label_smoothing=smoothing).numpy()
0.57618
```

After:

```
>>> cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
...   label_smoothing=smoothing)
>>> cce(y_true, y_pred, sample_weight=weights).numpy()
0.57618
```



 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

`weights` acts as a coefficient for the loss. If a scalar is provided,
then the loss is simply scaled by the given value. If `weights` is a
tensor of shape `[batch_size]`, then the loss weights apply to each
corresponding sample.

If `label_smoothing` is nonzero, smooth the labels towards 1/num_classes:
    new_onehot_labels = onehot_labels * (1 - label_smoothing)
                        + label_smoothing / num_classes

Note that `onehot_labels` and `logits` must have the same shape,
e.g. `[batch_size, num_classes]`. The shape of `weights` must be
broadcastable to loss, whose shape is decided by the shape of `logits`.
In case the shape of `logits` is `[batch_size, num_classes]`, loss is
a `Tensor` of shape `[batch_size]`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`onehot_labels`
</td>
<td>
One-hot-encoded labels.
</td>
</tr><tr>
<td>
`logits`
</td>
<td>
Logits outputs of the network.
</td>
</tr><tr>
<td>
`weights`
</td>
<td>
Optional `Tensor` that is broadcastable to loss.
</td>
</tr><tr>
<td>
`label_smoothing`
</td>
<td>
If greater than 0 then smooth the labels.
</td>
</tr><tr>
<td>
`scope`
</td>
<td>
the scope for the operations performed in computing the loss.
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
Weighted loss `Tensor` of the same type as `logits`. If `reduction` is
`NONE`, this has shape `[batch_size]`; otherwise, it is scalar.
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
If the shape of `logits` doesn't match that of `onehot_labels`
or if the shape of `weights` is invalid or if `weights` is None.  Also if
`onehot_labels` or `logits` is None.
</td>
</tr>
</table>


