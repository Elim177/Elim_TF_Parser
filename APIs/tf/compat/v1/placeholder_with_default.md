description: A placeholder op that passes through input when its output is not fed.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.placeholder_with_default" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.placeholder_with_default

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/array_ops.py">View source</a>



A placeholder op that passes through `input` when its output is not fed.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.placeholder_with_default(
    input, shape, name=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

This API is strongly discouraged for use with eager execution and
<a href="../../../tf/function.md"><code>tf.function</code></a>. The primary use of this API is for testing computation wrapped
within a <a href="../../../tf/function.md"><code>tf.function</code></a> where the input tensors might not have statically known
fully-defined shapes. The same can be achieved by creating a
[concrete function](
https://www.tensorflow.org/guide/function#obtaining_concrete_functions)
from the <a href="../../../tf/function.md"><code>tf.function</code></a> with a <a href="../../../tf/TensorSpec.md"><code>tf.TensorSpec</code></a> input which has partially
defined shapes. For example, the code

```
>>> @tf.function
... def f():
...   x = tf.compat.v1.placeholder_with_default(
...       tf.constant([[1., 2., 3.], [4., 5., 6.]]), [None, 3])
...   y = tf.constant([[1.],[2.], [3.]])
...   z = tf.matmul(x, y)
...   assert z.shape[0] == None
...   assert z.shape[1] == 1
```

```
>>> f()
```

can easily be replaced by

```
>>> @tf.function
... def f(x):
...   y = tf.constant([[1.],[2.], [3.]])
...   z = tf.matmul(x, y)
...   assert z.shape[0] == None
...   assert z.shape[1] == 1
```

```
>>> g = f.get_concrete_function(tf.TensorSpec([None, 3]))
```

You can learn more about <a href="../../../tf/function.md"><code>tf.function</code></a> at [Better
performance with tf.function](https://www.tensorflow.org/guide/function).


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`
</td>
<td>
A `Tensor`. The default value to produce when output is not fed.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
A <a href="../../../tf/TensorShape.md"><code>tf.TensorShape</code></a> or list of `int`s. The (possibly partial) shape of
the tensor.
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
A `Tensor`. Has the same type as `input`.
</td>
</tr>

</table>

