description: Types of loss reduction.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.losses.Reduction" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="all"/>
<meta itemprop="property" content="validate"/>
<meta itemprop="property" content="AUTO"/>
<meta itemprop="property" content="NONE"/>
<meta itemprop="property" content="SUM"/>
<meta itemprop="property" content="SUM_OVER_BATCH_SIZE"/>
</div>

# tf.keras.losses.Reduction

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/utils/losses_utils.py#L24-L84">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Types of loss reduction.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.losses.Reduction`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

Contains the following values:

* `AUTO`: Indicates that the reduction option will be determined by the usage
   context. For almost all cases this defaults to `SUM_OVER_BATCH_SIZE`. When
   used with <a href="../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a>, outside of built-in training loops such
   as <a href="../../../tf/keras.md"><code>tf.keras</code></a> `compile` and `fit`, we expect reduction value to be
   `SUM` or `NONE`. Using `AUTO` in that case will raise an error.
* `NONE`: No **additional** reduction is applied to the output of the wrapped
   loss function. When non-scalar losses are returned to Keras functions like
   `fit`/`evaluate`, the unreduced vector loss is passed to the optimizer
   but the reported loss will be a scalar value.

   Caution: **Verify the shape of the outputs when using** <a href="../../../tf/keras/losses/Reduction.md#NONE"><code>Reduction.NONE</code></a>.
   The builtin loss functions wrapped by the loss classes reduce
   one dimension (`axis=-1`, or `axis` if specified by loss function).
   <a href="../../../tf/keras/losses/Reduction.md#NONE"><code>Reduction.NONE</code></a> just means that no **additional** reduction is applied by
   the class wrapper. For categorical losses with an example input shape of
   `[batch, W, H, n_classes]` the `n_classes` dimension is reduced. For
   pointwise losses your must include a dummy axis so that `[batch, W, H, 1]`
   is reduced to `[batch, W, H]`. Without the dummy axis `[batch, W, H]`
   will be incorrectly reduced to `[batch, W]`.

* `SUM`: Scalar sum of weighted losses.
* `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.
   This reduction type is not supported when used with
   <a href="../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a> outside of built-in training loops like <a href="../../../tf/keras.md"><code>tf.keras</code></a>
   `compile`/`fit`.

   You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:
   ```
   with strategy.scope():
     loss_obj = tf.keras.losses.CategoricalCrossentropy(
         reduction=tf.keras.losses.Reduction.NONE)
     ....
     loss = tf.reduce_sum(loss_obj(labels, predictions)) *
         (1. / global_batch_size)
   ```

Please see the [custom training guide](
https://www.tensorflow.org/tutorials/distribute/custom_training) for more
details on this.

## Methods

<h3 id="all"><code>all</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/utils/losses_utils.py#L76-L78">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>all()
</code></pre>




<h3 id="validate"><code>validate</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/utils/losses_utils.py#L80-L84">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>validate(
    key
)
</code></pre>








<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
AUTO<a id="AUTO"></a>
</td>
<td>
`'auto'`
</td>
</tr><tr>
<td>
NONE<a id="NONE"></a>
</td>
<td>
`'none'`
</td>
</tr><tr>
<td>
SUM<a id="SUM"></a>
</td>
<td>
`'sum'`
</td>
</tr><tr>
<td>
SUM_OVER_BATCH_SIZE<a id="SUM_OVER_BATCH_SIZE"></a>
</td>
<td>
`'sum_over_batch_size'`
</td>
</tr>
</table>

