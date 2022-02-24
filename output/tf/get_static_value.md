description: Returns the constant value of the given tensor, if efficiently calculable.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.get_static_value" />
<meta itemprop="path" content="Stable" />
</div>

# tf.get_static_value

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_util.py">View source</a>



Returns the constant value of the given tensor, if efficiently calculable.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.get_static_value`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.get_static_value(
    tensor, partial=(False)
)
</code></pre>



<!-- Placeholder for "Used in" -->

This function attempts to partially evaluate the given tensor, and
returns its value as a numpy ndarray if this succeeds.

#### Example usage:



```
>>> a = tf.constant(10)
>>> tf.get_static_value(a)
10
>>> b = tf.constant(20)
>>> tf.get_static_value(tf.add(a, b))
30
```

```
>>> # `tf.Variable` is not supported.
>>> c = tf.Variable(30)
>>> print(tf.get_static_value(c))
None
```

Using `partial` option is most relevant when calling `get_static_value` inside
a <a href="../tf/function.md"><code>tf.function</code></a>. Setting it to `True` will return the results but for the
values that cannot be evaluated will be `None`. For example:

```python
class Foo(object):
  def __init__(self):
    self.a = tf.Variable(1)
    self.b = tf.constant(2)

  @tf.function
  def bar(self, partial):
    packed = tf.raw_ops.Pack(values=[self.a, self.b])
    static_val = tf.get_static_value(packed, partial=partial)
    tf.print(static_val)

f = Foo()
f.bar(partial=True)  # `array([None, array(2, dtype=int32)], dtype=object)`
f.bar(partial=False)  # `None`
```

Compatibility(V1): If `constant_value(tensor)` returns a non-`None` result, it
will no longer be possible to feed a different value for `tensor`. This allows
the result of this function to influence the graph that is constructed, and
permits static shape optimizations.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensor`
</td>
<td>
The Tensor to be evaluated.
</td>
</tr><tr>
<td>
`partial`
</td>
<td>
If True, the returned numpy array is allowed to have partially
evaluated values. Values that can't be evaluated will be None.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A numpy ndarray containing the constant value of the given `tensor`,
or None if it cannot be calculated.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
if tensor is not an ops.Tensor.
</td>
</tr>
</table>

