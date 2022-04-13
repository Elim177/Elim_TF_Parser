description: TensorFlow variant of NumPy's array.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.numpy.array" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.numpy.array

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/numpy_ops/np_array_ops.py">View source</a>



TensorFlow variant of NumPy's `array`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.numpy.array(
    val, dtype=None, copy=(True), ndmin=0
)
</code></pre>



<!-- Placeholder for "Used in" -->

Since Tensors are immutable, a copy is made only if val is placed on a

  different device than the current one. Even if `copy` is False, a new Tensor
  may need to be built to satisfy `dtype` and `ndim`. This is used only if `val`
  is an ndarray or a Tensor.
  

See the NumPy documentation for [`numpy.array`](https://numpy.org/doc/1.16/reference/generated/numpy.array.html).