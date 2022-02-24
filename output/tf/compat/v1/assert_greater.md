description: Assert the condition x > y holds element-wise.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.assert_greater" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.assert_greater

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/check_ops.py">View source</a>



Assert the condition `x > y` holds element-wise.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.debugging.assert_greater`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.assert_greater(
    x, y, data=None, summarize=None, message=None, name=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

<a href="../../../tf/compat/v1/assert_greater.md"><code>tf.compat.v1.assert_greater</code></a> is compatible with eager execution and
<a href="../../../tf/function.md"><code>tf.function</code></a>.
Please use <a href="../../../tf/debugging/assert_greater.md"><code>tf.debugging.assert_greater</code></a> instead when migrating to TF2. Apart
from `data`, all arguments are supported with the same argument name.

If you want to ensure the assert statements run before the
potentially-invalid computation, please use <a href="../../../tf/control_dependencies.md"><code>tf.control_dependencies</code></a>,
as tf.function auto-control dependencies are insufficient for assert
statements.

#### Structural Mapping to Native TF2

Before:

```python
tf.compat.v1.assert_greater(
  x=x, y=y, data=data, summarize=summarize,
  message=message, name=name)
```

After:

```python
tf.debugging.assert_greater(
  x=x, y=y, message=message,
  summarize=summarize, name=name)
```

#### TF1 & TF2 Usage Example

TF1:

```
>>> g = tf.Graph()
>>> with g.as_default():
...   a = tf.compat.v1.placeholder(tf.float32, [2])
...   b = tf.compat.v1.placeholder(tf.float32, [2])
...   result = tf.compat.v1.assert_greater(a, b,
...     message='"a > b" does not hold for the given inputs')
...   with tf.compat.v1.control_dependencies([result]):
...     sum_node = a + b
>>> sess = tf.compat.v1.Session(graph=g)
>>> val = sess.run(sum_node, feed_dict={a: [1, 2], b:[0, 1]})
```


TF2:

```
>>> a = tf.Variable([1, 2], dtype=tf.float32)
>>> b = tf.Variable([0, 1], dtype=tf.float32)
>>> assert_op = tf.debugging.assert_greater(a, b, message=
...   '"a > b" does not hold for the given inputs')
>>> # When working with tf.control_dependencies
>>> with tf.control_dependencies([assert_op]):
...   val = a + b
```



 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

This condition holds if for every pair of (possibly broadcast) elements
`x[i]`, `y[i]`, we have `x[i] > y[i]`.
If both `x` and `y` are empty, this is trivially satisfied.

When running in graph mode, you should add a dependency on this operation
to ensure that it runs. Example of adding a dependency to an operation:

```python
with tf.control_dependencies([tf.compat.v1.assert_greater(x, y)]):
  output = tf.reduce_sum(x)
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
 Numeric `Tensor`.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
 Numeric `Tensor`, same dtype as and broadcastable to `x`.
</td>
</tr><tr>
<td>
`data`
</td>
<td>
 The tensors to print out if the condition is False.  Defaults to
error message and first few entries of `x`, `y`.
</td>
</tr><tr>
<td>
`summarize`
</td>
<td>
Print this many entries of each tensor.
</td>
</tr><tr>
<td>
`message`
</td>
<td>
A string to prefix to the default message.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for this operation (optional).  Defaults to "assert_greater".
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Op that raises `InvalidArgumentError` if `x > y` is False.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`InvalidArgumentError`
</td>
<td>
if the check can be performed immediately and
`x > y` is False. The check can be performed immediately during
eager execution or if `x` and `y` are statically known.
</td>
</tr>
</table>


