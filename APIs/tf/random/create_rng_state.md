description: Creates a RNG state from an integer or a vector.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.random.create_rng_state" />
<meta itemprop="path" content="Stable" />
</div>

# tf.random.create_rng_state

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/stateful_random_ops.py">View source</a>



Creates a RNG state from an integer or a vector.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.random.experimental.create_rng_state`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.random.create_rng_state`, `tf.compat.v1.random.experimental.create_rng_state`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.random.create_rng_state(
    seed, alg
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Example:



```
>>> tf.random.create_rng_state(
...     1234, "philox")
<tf.Tensor: shape=(3,), dtype=int64, numpy=array([1234,    0,    0])>
>>> tf.random.create_rng_state(
...     [12, 34], "threefry")
<tf.Tensor: shape=(2,), dtype=int64, numpy=array([12, 34])>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`seed`
</td>
<td>
an integer or 1-D numpy array.
</td>
</tr><tr>
<td>
`alg`
</td>
<td>
the RNG algorithm. Can be a string, an `Algorithm` or an integer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
a 1-D numpy array whose size depends on the algorithm.
</td>
</tr>

</table>

