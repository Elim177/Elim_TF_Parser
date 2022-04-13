description: Update ref by assigning value to it.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.assign" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.assign

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/state_ops.py">View source</a>



Update `ref` by assigning `value` to it.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.assign(
    ref, value, validate_shape=None, use_locking=None, name=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

<a href="../../../tf/compat/v1/assign.md"><code>tf.compat.v1.assign</code></a> is mostly compatible with eager
execution and <a href="../../../tf/function.md"><code>tf.function</code></a>. However, argument 'validate_shape' will be
ignored. To avoid shape validation, set 'shape' to tf.TensorShape(None) when
constructing the variable:

```
>>> import tensorflow as tf
>>> a = tf.Variable([1], shape=tf.TensorShape(None))
>>> tf.compat.v1.assign(a, [2,3])
```

To switch to the native TF2 style, one could use method 'assign' of
<a href="../../../tf/Variable.md"><code>tf.Variable</code></a>:

#### How to Map Arguments

| TF1 Arg Name          | TF2 Arg Name    | Note                       |
| :-------------------- | :-------------- | :------------------------- |
| `ref`                 | `self`          | In `assign()` method       |
| `value`               | `value`         | In `assign()` method       |
| `validate_shape`      | Not supported   | Specify `shape` in the     |
:                       :                 : constructor to replicate   :
:                       :                 : behavior                   :
| `use_locking`         | `use_locking`   | In `assign()` method       |
| `name`                | `name`          | In `assign()` method       |
| -                     | `read_value`    | Set to True to replicate   |
:                       :                 : behavior (True is default) :


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

This operation outputs a Tensor that holds the new value of `ref` after
the value has been assigned. This makes it easier to chain operations that
need to use the reset value.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`ref`
</td>
<td>
A mutable `Tensor`. Should be from a `Variable` node. May be
uninitialized.
</td>
</tr><tr>
<td>
`value`
</td>
<td>
A `Tensor`. Must have the same shape and dtype as `ref`. The value to
be assigned to the variable.
</td>
</tr><tr>
<td>
`validate_shape`
</td>
<td>
An optional `bool`. Defaults to `True`. If true, the
operation will validate that the shape of 'value' matches the shape of the
Tensor being assigned to.  If false, 'ref' will take on the shape of
'value'.
</td>
</tr><tr>
<td>
`use_locking`
</td>
<td>
An optional `bool`. Defaults to `True`. If True, the assignment
will be protected by a lock; otherwise the behavior is undefined, but may
exhibit less contention.
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
A `Tensor` that will hold the new value of `ref` after
the assignment has completed.
</td>
</tr>

</table>


#### Before & After Usage Example

#### Before:



```
>>> with tf.Graph().as_default():
...   with tf.compat.v1.Session() as sess:
...     a = tf.compat.v1.Variable(0, dtype=tf.int64)
...     sess.run(a.initializer)
...     update_op = tf.compat.v1.assign(a, 2)
...     res_a = sess.run(update_op)
...     res_a
2
```

#### After:



```
>>> b = tf.Variable(0, dtype=tf.int64)
>>> res_b = b.assign(2)
>>> res_b.numpy()
2
```