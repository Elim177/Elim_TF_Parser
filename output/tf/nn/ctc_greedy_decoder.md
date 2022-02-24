description: Performs greedy decoding on the logits given in input (best path).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nn.ctc_greedy_decoder" />
<meta itemprop="path" content="Stable" />
</div>

# tf.nn.ctc_greedy_decoder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/ctc_ops.py">View source</a>



Performs greedy decoding on the logits given in input (best path).

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.nn.ctc_greedy_decoder`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.nn.ctc_greedy_decoder(
    inputs, sequence_length, merge_repeated=(True), blank_index=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Given a tensor as `inputs`, the `blank_index` parameter defines the class
index of the blank symbol.

#### For example:



If `blank_index` is equal to 1:

```
>>> inf = float("inf")
>>> logits = tf.constant([[[   0., -inf, -inf],
...                        [ -2.3, -inf, -0.1]],
...                       [[ -inf, -0.5, -inf],
...                        [ -inf, -inf, -0.1]],
...                       [[ -inf, -inf, -inf],
...                        [ -0.1, -inf, -2.3]]])
>>> seq_lens = tf.constant([2, 3])
>>> outputs = tf.nn.ctc_greedy_decoder(
...     logits,
...     seq_lens,
...     blank_index=1)
```

#### Notes:



- Regardless of the value of `merge_repeated`, if an index of a
  given time and batch corresponds to the `blank_index`, no new
  element is emitted.
- Default `blank_index` is `(num_classes - 1)`, unless overriden.

If `merge_repeated` is `True`, merge repeated classes in output.
This means that if consecutive logits' maximum indices are the same,
only the first of these is emitted.  The sequence `A B B * B * B` (where '*'
is the blank label) becomes

  * `A B B B` if `merge_repeated=True`.
  * `A B B B B` if `merge_repeated=False`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`inputs`
</td>
<td>
3-D `float` `Tensor` sized `[max_time, batch_size, num_classes]`.
The logits.
</td>
</tr><tr>
<td>
`sequence_length`
</td>
<td>
1-D `int32` vector containing sequence lengths, having size
`[batch_size]`.
</td>
</tr><tr>
<td>
`merge_repeated`
</td>
<td>
Boolean.  Default: True.
</td>
</tr><tr>
<td>
`blank_index`
</td>
<td>
(Optional). Default: `num_classes - 1`. Define the class index
to use for the blank label. Negative values will start from num_classes,
ie, -1 will reproduce the ctc_greedy_decoder behavior of using
num_classes - 1 for the blank symbol, which corresponds to the default.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tuple `(decoded, neg_sum_logits)` where
</td>
</tr>
<tr>
<td>
`decoded`
</td>
<td>
A single-element list. `decoded[0]`
is an `SparseTensor` containing the decoded outputs s.t.:

`decoded.indices`: Indices matrix `(total_decoded_outputs, 2)`.
  The rows store: `[batch, time]`.

`decoded.values`: Values vector, size `(total_decoded_outputs)`.
  The vector stores the decoded classes.

`decoded.dense_shape`: Shape vector, size `(2)`.
  The shape values are: `[batch_size, max_decoded_length]`
</td>
</tr><tr>
<td>
`neg_sum_logits`
</td>
<td>
A `float` matrix `(batch_size x 1)` containing, for the
sequence found, the negative of the sum of the greatest logit at each
timeframe.
</td>
</tr>
</table>

