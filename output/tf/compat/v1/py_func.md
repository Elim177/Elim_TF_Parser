description: Wraps a python function and uses it as a TensorFlow op.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.py_func" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.py_func

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/script_ops.py">View source</a>



Wraps a python function and uses it as a TensorFlow op.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.py_func(
    func, inp, Tout, stateful=(True), name=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

This name was deprecated and removed in TF2, but <a href="../../../tf/numpy_function.md"><code>tf.numpy_function</code></a> is a
near-exact replacement, just drop the `stateful` argument (all
<a href="../../../tf/numpy_function.md"><code>tf.numpy_function</code></a> calls are considered stateful). It is compatible with
eager execution and <a href="../../../tf/function.md"><code>tf.function</code></a>.

<a href="../../../tf/py_function.md"><code>tf.py_function</code></a> is a close but not an exact replacement, passing TensorFlow
tensors to the wrapped function instead of NumPy arrays, which provides
gradients and can take advantage of accelerators.

Before:

```
>>> def fn_using_numpy(x):
...   x[0] = 0.
...   return x
>>> tf.compat.v1.py_func(fn_using_numpy, inp=[tf.constant([1., 2.])],
...     Tout=tf.float32, stateful=False)
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([0., 2.], dtype=float32)>
```

After:

```
>>> tf.numpy_function(fn_using_numpy, inp=[tf.constant([1., 2.])],
...     Tout=tf.float32)
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([0., 2.], dtype=float32)>
```



 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

Given a python function `func`, which takes numpy arrays as its
arguments and returns numpy arrays as its outputs, wrap this function as an
operation in a TensorFlow graph. The following snippet constructs a simple
TensorFlow graph that invokes the `np.sinh()` NumPy function as a operation
in the graph:

```python
def my_func(x):
  # x will be a numpy array with the contents of the placeholder below
  return np.sinh(x)
input = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.py_func(my_func, [input], tf.float32)
```

**N.B.** The <a href="../../../tf/compat/v1/py_func.md"><code>tf.compat.v1.py_func()</code></a> operation has the following known
limitations:

* The body of the function (i.e. `func`) will not be serialized in a
  `GraphDef`. Therefore, you should not use this function if you need to
  serialize your model and restore it in a different environment.

* The operation must run in the same address space as the Python program
  that calls <a href="../../../tf/compat/v1/py_func.md"><code>tf.compat.v1.py_func()</code></a>. If you are using distributed
  TensorFlow, you
  must run a <a href="../../../tf/distribute/Server.md"><code>tf.distribute.Server</code></a> in the same process as the program that
  calls
  <a href="../../../tf/compat/v1/py_func.md"><code>tf.compat.v1.py_func()</code></a> and you must pin the created operation to a device
  in that
  server (e.g. using `with tf.device():`).

Note: It produces tensors of unknown shape and rank as shape inference
  does not work on arbitrary Python code.
  If you need the shape, you need to set it based on statically
  available information.

  E.g.
  ```python
  import tensorflow as tf
  import numpy as np

  def make_synthetic_data(i):
      return np.cast[np.uint8](i) * np.ones([20,256,256,3],
              dtype=np.float32) / 10.

  def preprocess_fn(i):
      ones = tf.py_function(make_synthetic_data,[i],tf.float32)
      ones.set_shape(tf.TensorShape([None, None, None, None]))
      ones = tf.image.resize(ones, [224,224])
      return ones

  ds = tf.data.Dataset.range(10)
  ds = ds.map(preprocess_fn)
  ```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`func`
</td>
<td>
A Python function, which accepts `ndarray` objects as arguments and
returns a list of `ndarray` objects (or a single `ndarray`). This function
must accept as many arguments as there are tensors in `inp`, and these
argument types will match the corresponding <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a> objects in `inp`.
The returns `ndarray`s must match the number and types defined `Tout`.
Important Note: Input and output numpy `ndarray`s of `func` are not
  guaranteed to be copies. In some cases their underlying memory will be
  shared with the corresponding TensorFlow tensors. In-place modification
  or storing `func` input or return values in python datastructures
  without explicit (np.)copy can have non-deterministic consequences.
</td>
</tr><tr>
<td>
`inp`
</td>
<td>
A list of `Tensor` objects.
</td>
</tr><tr>
<td>
`Tout`
</td>
<td>
A list or tuple of tensorflow data types or a single tensorflow data
type if there is only one, indicating what `func` returns.
</td>
</tr><tr>
<td>
`stateful`
</td>
<td>
(Boolean.) If True, the function should be considered stateful. If
a function is stateless, when given the same input it will return the same
output and have no observable side effects. Optimizations such as common
subexpression elimination are only performed on stateless operations.
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
A list of `Tensor` or a single `Tensor` which `func` computes.
</td>
</tr>

</table>


