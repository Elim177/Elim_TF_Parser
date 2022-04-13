description: Compute the matrix rank of one or more matrices.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.linalg.matrix_rank" />
<meta itemprop="path" content="Stable" />
</div>

# tf.linalg.matrix_rank

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/linalg/linalg_impl.py">View source</a>



Compute the matrix rank of one or more matrices.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.linalg.matrix_rank`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.linalg.matrix_rank(
    a, tol=None, validate_args=(False), name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`a`
</td>
<td>
(Batch of) `float`-like matrix-shaped `Tensor`(s) which are to be
pseudo-inverted.
</td>
</tr><tr>
<td>
`tol`
</td>
<td>
Threshold below which the singular value is counted as 'zero'.
Default value: `None` (i.e., `eps * max(rows, cols) * max(singular_val)`).
</td>
</tr><tr>
<td>
`validate_args`
</td>
<td>
When `True`, additional assertions might be embedded in the
graph.
Default value: `False` (i.e., no graph assertions are added).
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Python `str` prefixed to ops created by this function.
Default value: 'matrix_rank'.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`matrix_rank`
</td>
<td>
(Batch of) `int32` scalars representing the number of non-zero
singular values.
</td>
</tr>
</table>

