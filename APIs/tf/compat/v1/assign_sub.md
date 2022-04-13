description: Update ref by subtracting value from it.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.assign_sub" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.assign_sub

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/state_ops.py">View source</a>



Update `ref` by subtracting `value` from it.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.assign_sub(
    ref, value, use_locking=None, name=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

<a href="../../../tf/compat/v1/assign_sub.md"><code>tf.compat.v1.assign_sub</code></a> is mostly compatible with eager
execution and <a href="../../../tf/function.md"><code>tf.function</code></a>.

To switch to the native TF2 style, one could use method 'assign_sub' of
<a href="../../../tf/Variable.md"><code>tf.Variable</code></a>:

#### How to Map Arguments

| TF1 Arg Name          | TF2 Arg Name    | Note                       |
| :-------------------- | :-------------- | :------------------------- |
| `ref`                 | `self`          | In `assign_sub()` method   |
| `value`               | `value`         | In `assign_sub()` method   |
| `use_locking`         | `use_locking`   | In `assign_sub()` method   |
| `name`                | `name`          | In `assign_sub()` method   |
| -                     | `read_value`    | Set to True to replicate   |
:                       :                 : behavior (True is default) :


#### Before & After Usage Example

Before:

```
>>> with tf.Graph().as_default():
...   with tf.compat.v1.Session() as sess:
...     a = tf.compat.v1.Variable(1, dtype=tf.int64)
...     sess.run(a.initializer)
...     update_op = tf.compat.v1.assign_sub(a, 1)
...     res_a = sess.run(update_op)
...     res_a
0
```

After:

```
>>> b = tf.Variable(1, dtype=tf.int64)
>>> res_b = b.assign_sub(1)
>>> res_b.numpy()
0
```



 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.
Unlike <a href="../../../tf/math/subtract.md"><code>tf.math.subtract</code></a>, this op does not broadcast. `ref` and `value`
must have the same shape.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`ref`
</td>
<td>
A mutable `Tensor`. Must be one of the following types: `float32`,
`float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`,
`complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`. Should be
from a `Variable` node.
</td>
</tr><tr>
<td>
`value`
</td>
<td>
A `Tensor`. Must have the same shape and dtype as `ref`. The value to
be subtracted to the variable.
</td>
</tr><tr>
<td>
`use_locking`
</td>
<td>
An optional `bool`. Defaults to `False`. If True, the
subtraction will be protected by a lock; otherwise the behavior is
undefined, but may exhibit less contention.
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
Same as `ref`.  Returned as a convenience for operations that want
to use the new value after the variable has been updated.
</td>
</tr>

</table>


