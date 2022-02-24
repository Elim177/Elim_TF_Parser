description: Decorator to override default implementation for binary elementwise APIs.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dispatch_for_binary_elementwise_apis" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dispatch_for_binary_elementwise_apis

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/util/dispatch.py">View source</a>



Decorator to override default implementation for binary elementwise APIs.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.experimental.dispatch_for_binary_elementwise_apis`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dispatch_for_binary_elementwise_apis(
    x_type, y_type
)
</code></pre>



<!-- Placeholder for "Used in" -->

The decorated function (known as the "elementwise api handler") overrides
the default implementation for any binary elementwise API whenever the value
for the first two arguments (typically named `x` and `y`) match the specified
type annotations.  The elementwise api handler is called with two arguments:

  `elementwise_api_handler(api_func, x, y)`

Where `x` and `y` are the first two arguments to the elementwise api, and
`api_func` is a TensorFlow function that takes two parameters and performs the
elementwise operation (e.g., <a href="../../tf/math/add.md"><code>tf.add</code></a>).

The following example shows how this decorator can be used to update all
binary elementwise operations to handle a `MaskedTensor` type:

```
>>> from tensorflow.python.framework import extension_type
>>> class MaskedTensor(extension_type.ExtensionType):
...   values: tf.Tensor
...   mask: tf.Tensor
>>> @dispatch_for_binary_elementwise_apis(MaskedTensor, MaskedTensor)
... def binary_elementwise_api_handler(api_func, x, y):
...   return MaskedTensor(api_func(x.values, y.values), x.mask & y.mask)
>>> a = MaskedTensor([1, 2, 3, 4, 5], [True, True, True, True, False])
>>> b = MaskedTensor([2, 4, 6, 8, 0], [True, True, True, False, True])
>>> c = tf.add(a, b)
>>> print(f"values={c.values.numpy()}, mask={c.mask.numpy()}")
values=[ 3 6 9 12 5], mask=[ True True True False False]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x_type`
</td>
<td>
A type annotation indicating when the api handler should be called.
</td>
</tr><tr>
<td>
`y_type`
</td>
<td>
A type annotation indicating when the api handler should be called.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A decorator.
</td>
</tr>

</table>


#### Registered APIs

The binary elementwise APIs are:

* <a href="../../tf/bitwise/bitwise_and.md"><code>tf.bitwise.bitwise_and(x, y, name=None)</code></a>
* <a href="../../tf/bitwise/bitwise_or.md"><code>tf.bitwise.bitwise_or(x, y, name=None)</code></a>
* <a href="../../tf/bitwise/bitwise_xor.md"><code>tf.bitwise.bitwise_xor(x, y, name=None)</code></a>
* <a href="../../tf/bitwise/left_shift.md"><code>tf.bitwise.left_shift(x, y, name=None)</code></a>
* <a href="../../tf/bitwise/right_shift.md"><code>tf.bitwise.right_shift(x, y, name=None)</code></a>
* <a href="../../tf/dtypes/complex.md"><code>tf.dtypes.complex(real, imag, name=None)</code></a>
* <a href="../../tf/math/add.md"><code>tf.math.add(x, y, name=None)</code></a>
* <a href="../../tf/math/atan2.md"><code>tf.math.atan2(y, x, name=None)</code></a>
* <a href="../../tf/math/divide.md"><code>tf.math.divide(x, y, name=None)</code></a>
* <a href="../../tf/math/divide_no_nan.md"><code>tf.math.divide_no_nan(x, y, name=None)</code></a>
* <a href="../../tf/math/equal.md"><code>tf.math.equal(x, y, name=None)</code></a>
* <a href="../../tf/math/floordiv.md"><code>tf.math.floordiv(x, y, name=None)</code></a>
* <a href="../../tf/math/floormod.md"><code>tf.math.floormod(x, y, name=None)</code></a>
* <a href="../../tf/math/greater.md"><code>tf.math.greater(x, y, name=None)</code></a>
* <a href="../../tf/math/greater_equal.md"><code>tf.math.greater_equal(x, y, name=None)</code></a>
* <a href="../../tf/math/less.md"><code>tf.math.less(x, y, name=None)</code></a>
* <a href="../../tf/math/less_equal.md"><code>tf.math.less_equal(x, y, name=None)</code></a>
* <a href="../../tf/math/logical_and.md"><code>tf.math.logical_and(x, y, name=None)</code></a>
* <a href="../../tf/math/logical_or.md"><code>tf.math.logical_or(x, y, name=None)</code></a>
* `tf.math.logical_xor(x, y, name='LogicalXor')`
* <a href="../../tf/math/maximum.md"><code>tf.math.maximum(x, y, name=None)</code></a>
* <a href="../../tf/math/minimum.md"><code>tf.math.minimum(x, y, name=None)</code></a>
* <a href="../../tf/math/multiply.md"><code>tf.math.multiply(x, y, name=None)</code></a>
* <a href="../../tf/math/multiply_no_nan.md"><code>tf.math.multiply_no_nan(x, y, name=None)</code></a>
* <a href="../../tf/math/not_equal.md"><code>tf.math.not_equal(x, y, name=None)</code></a>
* <a href="../../tf/math/pow.md"><code>tf.math.pow(x, y, name=None)</code></a>
* <a href="../../tf/math/squared_difference.md"><code>tf.math.squared_difference(x, y, name=None)</code></a>
* <a href="../../tf/math/subtract.md"><code>tf.math.subtract(x, y, name=None)</code></a>
* <a href="../../tf/math/truediv.md"><code>tf.math.truediv(x, y, name=None)</code></a>
* <a href="../../tf/realdiv.md"><code>tf.realdiv(x, y, name=None)</code></a>
* <a href="../../tf/truncatediv.md"><code>tf.truncatediv(x, y, name=None)</code></a>
* <a href="../../tf/truncatemod.md"><code>tf.truncatemod(x, y, name=None)</code></a>