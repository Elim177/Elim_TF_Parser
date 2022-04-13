description: Computes dropout: randomly sets elements to zero to prevent overfitting.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nn.experimental.stateless_dropout" />
<meta itemprop="path" content="Stable" />
</div>

# tf.nn.experimental.stateless_dropout

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/nn_ops.py">View source</a>



Computes dropout: randomly sets elements to zero to prevent overfitting.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.nn.experimental.stateless_dropout`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.nn.experimental.stateless_dropout(
    x, rate, seed, rng_alg=None, noise_shape=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

[Dropout](https://arxiv.org/abs/1207.0580) is useful for regularizing DNN
models. Inputs elements are randomly set to zero (and the other elements are
rescaled). This encourages each node to be independently useful, as it cannot
rely on the output of other nodes.

More precisely: With probability `rate` elements of `x` are set to `0`.
The remaining elements are scaled up by `1.0 / (1 - rate)`, so that the
expected value is preserved.

```
>>> x = tf.ones([3,5])
>>> tf.nn.experimental.stateless_dropout(x, rate=0.5, seed=[1, 0])
<tf.Tensor: shape=(3, 5), dtype=float32, numpy=
array([[2., 0., 2., 0., 0.],
       [0., 0., 2., 0., 2.],
       [0., 0., 0., 0., 2.]], dtype=float32)>
```

```
>>> x = tf.ones([3,5])
>>> tf.nn.experimental.stateless_dropout(x, rate=0.8, seed=[1, 0])
<tf.Tensor: shape=(3, 5), dtype=float32, numpy=
array([[5., 0., 0., 0., 0.],
       [0., 0., 0., 0., 5.],
       [0., 0., 0., 0., 5.]], dtype=float32)>
```

```
>>> tf.nn.experimental.stateless_dropout(x, rate=0.0, seed=[1, 0]) == x
<tf.Tensor: shape=(3, 5), dtype=bool, numpy=
array([[ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True]])>
```


This function is a stateless version of <a href="../../../tf/nn/dropout.md"><code>tf.nn.dropout</code></a>, in the
sense that no matter how many times you call this function, the same
`seed` will lead to the same results, and different `seed` will lead
to different results.

```
>>> x = tf.ones([3,5])
>>> tf.nn.experimental.stateless_dropout(x, rate=0.8, seed=[1, 0])
<tf.Tensor: shape=(3, 5), dtype=float32, numpy=
array([[5., 0., 0., 0., 0.],
       [0., 0., 0., 0., 5.],
       [0., 0., 0., 0., 5.]], dtype=float32)>
>>> tf.nn.experimental.stateless_dropout(x, rate=0.8, seed=[1, 0])
<tf.Tensor: shape=(3, 5), dtype=float32, numpy=
array([[5., 0., 0., 0., 0.],
       [0., 0., 0., 0., 5.],
       [0., 0., 0., 0., 5.]], dtype=float32)>
>>> tf.nn.experimental.stateless_dropout(x, rate=0.8, seed=[2, 0])
<tf.Tensor: shape=(3, 5), dtype=float32, numpy=
array([[5., 0., 0., 0., 0.],
       [0., 0., 0., 5., 0.],
       [0., 0., 0., 0., 0.]], dtype=float32)>
>>> tf.nn.experimental.stateless_dropout(x, rate=0.8, seed=[2, 0])
<tf.Tensor: shape=(3, 5), dtype=float32, numpy=
array([[5., 0., 0., 0., 0.],
       [0., 0., 0., 5., 0.],
       [0., 0., 0., 0., 0.]], dtype=float32)>
```

Compare the above results to those of <a href="../../../tf/nn/dropout.md"><code>tf.nn.dropout</code></a> below. The
second time <a href="../../../tf/nn/dropout.md"><code>tf.nn.dropout</code></a> is called with the same seed, it will
give a different output.

```
>>> tf.random.set_seed(0)
>>> x = tf.ones([3,5])
>>> tf.nn.dropout(x, rate=0.8, seed=1)
<tf.Tensor: shape=(3, 5), dtype=float32, numpy=
array([[0., 0., 0., 5., 5.],
       [0., 5., 0., 5., 0.],
       [5., 0., 5., 0., 5.]], dtype=float32)>
>>> tf.nn.dropout(x, rate=0.8, seed=1)
<tf.Tensor: shape=(3, 5), dtype=float32, numpy=
array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 5., 0.],
       [0., 0., 0., 0., 0.]], dtype=float32)>
>>> tf.nn.dropout(x, rate=0.8, seed=2)
<tf.Tensor: shape=(3, 5), dtype=float32, numpy=
array([[0., 0., 0., 0., 0.],
       [0., 5., 0., 5., 0.],
       [0., 0., 0., 0., 0.]], dtype=float32)>
>>> tf.nn.dropout(x, rate=0.8, seed=2)
<tf.Tensor: shape=(3, 5), dtype=float32, numpy=
array([[0., 0., 0., 0., 0.],
       [5., 0., 5., 0., 5.],
       [0., 5., 0., 0., 5.]], dtype=float32)>
```

The difference between this function and <a href="../../../tf/nn/dropout.md"><code>tf.nn.dropout</code></a> is
analogous to the difference between <a href="../../../tf/random/stateless_uniform.md"><code>tf.random.stateless_uniform</code></a>
and <a href="../../../tf/random/uniform.md"><code>tf.random.uniform</code></a>. Please see [Random number
generation](https://www.tensorflow.org/guide/random_numbers) guide
for a detailed description of the various RNG systems in TF. As the
guide states, legacy stateful RNG ops like <a href="../../../tf/random/uniform.md"><code>tf.random.uniform</code></a> and
<a href="../../../tf/nn/dropout.md"><code>tf.nn.dropout</code></a> are not deprecated yet but highly discouraged,
because their states are hard to control.

By default, each element is kept or dropped independently.  If `noise_shape`
is specified, it must be
[broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
will make independent decisions. This is useful for dropping whole
channels from an image or sequence. For example:

```
>>> x = tf.ones([3,10])
>>> tf.nn.experimental.stateless_dropout(x, rate=2/3, noise_shape=[1,10],
...                                      seed=[1, 0])
<tf.Tensor: shape=(3, 10), dtype=float32, numpy=
array([[3., 0., 0., 0., 0., 0., 0., 3., 0., 3.],
       [3., 0., 0., 0., 0., 0., 0., 3., 0., 3.],
       [3., 0., 0., 0., 0., 0., 0., 3., 0., 3.]], dtype=float32)>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`
</td>
<td>
A floating point tensor.
</td>
</tr><tr>
<td>
`rate`
</td>
<td>
A scalar `Tensor` with the same type as x. The probability
that each element is dropped. For example, setting rate=0.1 would drop
10% of input elements.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
An integer tensor of shape `[2]`. The seed of the random numbers.
</td>
</tr><tr>
<td>
`rng_alg`
</td>
<td>
The algorithm used to generate the random numbers
(default to `"auto_select"`). See the `alg` argument of
<a href="../../../tf/random/stateless_uniform.md"><code>tf.random.stateless_uniform</code></a> for the supported values.
</td>
</tr><tr>
<td>
`noise_shape`
</td>
<td>
A 1-D integer `Tensor`, representing the
shape for randomly generated keep/drop flags.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for this operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Tensor of the same shape and dtype of `x`.
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
If `rate` is not in `[0, 1)` or if `x` is not a floating point
tensor. `rate=1` is disallowed, because the output would be all zeros,
which is likely not what was intended.
</td>
</tr>
</table>

