description: TensorFlow variant of NumPy's random.randn.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.numpy.random.randn" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.numpy.random.randn

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/numpy_ops/np_random.py">View source</a>



TensorFlow variant of NumPy's `random.randn`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.numpy.random.randn(
    *args
)
</code></pre>



<!-- Placeholder for "Used in" -->

Returns samples from a normal distribution.

  Uses `tf.random_normal`.

  Args:
    *args: The shape of the output array.

  Returns:
    An ndarray with shape `args` and dtype `float64`.
  

See the NumPy documentation for [`numpy.random.randn`](https://numpy.org/doc/1.16/reference/generated/numpy.random.randn.html).