description: Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.unstack" />
<meta itemprop="path" content="Stable" />
</div>

# tf.unstack

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/array_ops.py">View source</a>



Unpacks the given dimension of a rank-`R` tensor into rank-`(R-1)` tensors.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.unstack`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.unstack(
    value, num=None, axis=0, name=&#x27;unstack&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

Unpacks tensors from `value` by chipping it along the `axis` dimension.

```
>>> x = tf.reshape(tf.range(12), (3,4))
>>>
>>> p, q, r = tf.unstack(x)
>>> p.shape.as_list()
[4]
```

```
>>> i, j, k, l = tf.unstack(x, axis=1)
>>> i.shape.as_list()
[3]
```

This is the opposite of stack.

```
>>> x = tf.stack([i, j, k, l], axis=1)
```

More generally if you have a tensor of shape `(A, B, C, D)`:

```
>>> A, B, C, D = [2, 3, 4, 5]
>>> t = tf.random.normal(shape=[A, B, C, D])
```

The number of tensor returned is equal to the length of the target `axis`:

```
>>> axis = 2
>>> items = tf.unstack(t, axis=axis)
>>> len(items) == t.shape[axis]
True
```

The shape of each result tensor is equal to the shape of the input tensor,
with the target `axis` removed.

```
>>> items[0].shape.as_list()  # [A, B, D]
[2, 3, 5]
```

The value of each tensor `items[i]` is equal to the slice of `input` across
`axis` at index `i`:

```
>>> for i in range(len(items)):
...   slice = t[:,:,i,:]
...   assert tf.reduce_all(slice == items[i])
```

#### Python iterable unpacking

With eager execution you _can_ unstack the 0th axis of a tensor using python's
iterable unpacking:

```
>>> t = tf.constant([1,2,3])
>>> a,b,c = t
```

`unstack` is still necessary because Iterable unpacking doesn't work in
a <a href="../tf/function.md"><code>@tf.function</code></a>: Symbolic tensors are not iterable.

You need to use <a href="../tf/unstack.md"><code>tf.unstack</code></a> here:

```
>>> @tf.function
... def bad(t):
...   a,b,c = t
...   return a
>>>
>>> bad(t)
Traceback (most recent call last):
...
OperatorNotAllowedInGraphError: ...
```

```
>>> @tf.function
... def good(t):
...   a,b,c = tf.unstack(t)
...   return a
>>>
>>> good(t).numpy()
1
```

#### Unknown shapes

Eager tensors have concrete values, so their shape is always known.
Inside a <a href="../tf/function.md"><code>tf.function</code></a> the symbolic tensors may have unknown shapes.
If the length of `axis` is unknown <a href="../tf/unstack.md"><code>tf.unstack</code></a> will fail because it cannot
handle an unknown number of tensors:

```
>>> @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
... def bad(t):
...   tensors = tf.unstack(t)
...   return tensors[0]
>>>
>>> bad(tf.constant([1,2,3]))
Traceback (most recent call last):
...
ValueError: Cannot infer argument `num` from shape (None,)
```

If you know the `axis` length you can pass it as the `num` argument. But this
must be a constant value.

If you actually need a variable number of tensors in a single <a href="../tf/function.md"><code>tf.function</code></a>
trace, you will need to use exlicit loops and a <a href="../tf/TensorArray.md"><code>tf.TensorArray</code></a> instead.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`value`
</td>
<td>
A rank `R > 0` `Tensor` to be unstacked.
</td>
</tr><tr>
<td>
`num`
</td>
<td>
An `int`. The length of the dimension `axis`. Automatically inferred if
`None` (the default).
</td>
</tr><tr>
<td>
`axis`
</td>
<td>
An `int`. The axis to unstack along. Defaults to the first dimension.
Negative values wrap around, so the valid range is `[-R, R)`.
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
The list of `Tensor` objects unstacked from `value`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If `axis` is out of the range `[-R, R)`.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If `num` is unspecified and cannot be inferred.
</td>
</tr><tr>
<td>
`InvalidArgumentError`
</td>
<td>
If `num` does not match the shape of `value`.
</td>
</tr>
</table>

