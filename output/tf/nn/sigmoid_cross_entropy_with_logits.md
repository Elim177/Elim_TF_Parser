description: Computes sigmoid cross entropy given logits.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nn.sigmoid_cross_entropy_with_logits" />
<meta itemprop="path" content="Stable" />
</div>

# tf.nn.sigmoid_cross_entropy_with_logits

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/nn_impl.py">View source</a>



Computes sigmoid cross entropy given `logits`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.nn.sigmoid_cross_entropy_with_logits(
    labels=None, logits=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Measures the probability error in tasks with two outcomes in which each
outcome is independent and need not have a fully certain label. For instance,
one could perform a regression where the probability of an event happening is
known and used as a label. This loss may also be used for binary
classification, where labels are either zero or one.

For brevity, let `x = logits`, `z = labels`.  The logistic loss is

      z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
    = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
    = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
    = (1 - z) * x + log(1 + exp(-x))
    = x - x * z + log(1 + exp(-x))

For x < 0, to avoid overflow in exp(-x), we reformulate the above

      x - x * z + log(1 + exp(-x))
    = log(exp(x)) - x * z + log(1 + exp(-x))
    = - x * z + log(1 + exp(x))

Hence, to ensure stability and avoid overflow, the implementation uses this
equivalent formulation

    max(x, 0) - x * z + log(1 + exp(-abs(x)))

`logits` and `labels` must have the same type and shape.

```
>>> logits = tf.constant([1., -1., 0., 1., -1., 0., 0.])
>>> labels = tf.constant([0., 0., 0., 1., 1., 1., 0.5])
>>> tf.nn.sigmoid_cross_entropy_with_logits(
...     labels=labels, logits=logits).numpy()
array([1.3132617, 0.3132617, 0.6931472, 0.3132617, 1.3132617, 0.6931472,
       0.6931472], dtype=float32)
```

Compared to the losses which handle multiple outcomes,
<a href="../../tf/nn/softmax_cross_entropy_with_logits.md"><code>tf.nn.softmax_cross_entropy_with_logits</code></a> for general multi-class
classification and <a href="../../tf/nn/sparse_softmax_cross_entropy_with_logits.md"><code>tf.nn.sparse_softmax_cross_entropy_with_logits</code></a> for more
efficient multi-class classification with hard labels,
`sigmoid_cross_entropy_with_logits` is a slight simplification for binary
classification:

      sigmoid(x) = softmax([x, 0])[0]

$$\frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + e^0}$$

While `sigmoid_cross_entropy_with_logits` works for soft binary labels
(probabilities between 0 and 1), it can also be used for binary classification
where the labels are hard. There is an equivalence between all three symbols
in this case, with a probability 0 indicating the second class or 1 indicating
the first class:

```
>>> sigmoid_logits = tf.constant([1., -1., 0.])
>>> softmax_logits = tf.stack([sigmoid_logits, tf.zeros_like(sigmoid_logits)],
...                           axis=-1)
>>> soft_binary_labels = tf.constant([1., 1., 0.])
>>> soft_multiclass_labels = tf.stack(
...     [soft_binary_labels, 1. - soft_binary_labels], axis=-1)
>>> hard_labels = tf.constant([0, 0, 1])
>>> tf.nn.sparse_softmax_cross_entropy_with_logits(
...     labels=hard_labels, logits=softmax_logits).numpy()
array([0.31326166, 1.3132616 , 0.6931472 ], dtype=float32)
>>> tf.nn.softmax_cross_entropy_with_logits(
...     labels=soft_multiclass_labels, logits=softmax_logits).numpy()
array([0.31326166, 1.3132616, 0.6931472], dtype=float32)
>>> tf.nn.sigmoid_cross_entropy_with_logits(
...     labels=soft_binary_labels, logits=sigmoid_logits).numpy()
array([0.31326166, 1.3132616, 0.6931472], dtype=float32)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`labels`
</td>
<td>
A `Tensor` of the same type and shape as `logits`. Between 0 and 1,
inclusive.
</td>
</tr><tr>
<td>
`logits`
</td>
<td>
A `Tensor` of type `float32` or `float64`. Any real number.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` of the same shape as `logits` with the componentwise
logistic losses.
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
If `logits` and `labels` do not have the same shape.
</td>
</tr>
</table>

