description: Enables debug mode for tf.data.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.enable_debug_mode" />
<meta itemprop="path" content="Stable" />
</div>

# tf.data.experimental.enable_debug_mode

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/data/ops/dataset_ops.py">View source</a>



Enables debug mode for tf.data.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.data.experimental.enable_debug_mode`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.enable_debug_mode()
</code></pre>



<!-- Placeholder for "Used in" -->

Example usage with pdb module:
```
import tensorflow as tf
import pdb

tf.data.experimental.enable_debug_mode()

def func(x):
  # Python 3.7 and older requires `pdb.Pdb(nosigint=True).set_trace()`
  pdb.set_trace()
  x = x + 1
  return x

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
dataset = dataset.map(func)

for item in dataset:
  print(item)
```

The effect of debug mode is two-fold:

1) Any transformations that would introduce asynchrony, parallelism, or
non-determinism to the input pipeline execution will be forced to execute
synchronously, sequentially, and deterministically.

2) Any user-defined functions passed into tf.data transformations such as
`map` will be wrapped in <a href="../../../tf/py_function.md"><code>tf.py_function</code></a> so that their body is executed
"eagerly" as a Python function as opposed to a traced TensorFlow graph, which
is the default behavior. Note that even when debug mode is enabled, the
user-defined function is still traced  to infer the shape and type of its
outputs; as a consequence, any `print` statements or breakpoints will be
triggered once during the tracing before the actual execution of the input
pipeline.

NOTE: As the debug mode setting affects the construction of the tf.data input
pipeline, it should be enabled before any tf.data definitions.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
When invoked from graph mode.
</td>
</tr>
</table>

