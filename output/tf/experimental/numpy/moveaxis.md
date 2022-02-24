description: TensorFlow variant of NumPy's moveaxis.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.numpy.moveaxis" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.numpy.moveaxis

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/numpy_ops/np_array_ops.py">View source</a>



TensorFlow variant of NumPy's `moveaxis`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.numpy.moveaxis(
    a, source, destination
)
</code></pre>



<!-- Placeholder for "Used in" -->

Raises ValueError if source, destination not in (-ndim(a), ndim(a)).

See the NumPy documentation for [`numpy.moveaxis`](https://numpy.org/doc/1.16/reference/generated/numpy.moveaxis.html).