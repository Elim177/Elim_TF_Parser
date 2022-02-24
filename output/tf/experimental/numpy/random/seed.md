description: TensorFlow variant of NumPy's random.seed.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.numpy.random.seed" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.numpy.random.seed

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/numpy_ops/np_random.py">View source</a>



TensorFlow variant of NumPy's `random.seed`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.numpy.random.seed(
    s
)
</code></pre>



<!-- Placeholder for "Used in" -->

Sets the seed for the random number generator.

  Uses `tf.set_random_seed`.

  Args:
    s: an integer.
  

See the NumPy documentation for [`numpy.random.seed`](https://numpy.org/doc/1.16/reference/generated/numpy.random.seed.html).