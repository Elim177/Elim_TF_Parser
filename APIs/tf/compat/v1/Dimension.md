description: Represents the value of one dimension in a TensorShape.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.Dimension" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__bool__"/>
<meta itemprop="property" content="__div__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__floordiv__"/>
<meta itemprop="property" content="__ge__"/>
<meta itemprop="property" content="__gt__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__le__"/>
<meta itemprop="property" content="__lt__"/>
<meta itemprop="property" content="__mod__"/>
<meta itemprop="property" content="__mul__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__radd__"/>
<meta itemprop="property" content="__rdiv__"/>
<meta itemprop="property" content="__rfloordiv__"/>
<meta itemprop="property" content="__rmod__"/>
<meta itemprop="property" content="__rmul__"/>
<meta itemprop="property" content="__rsub__"/>
<meta itemprop="property" content="__rtruediv__"/>
<meta itemprop="property" content="__sub__"/>
<meta itemprop="property" content="__truediv__"/>
<meta itemprop="property" content="assert_is_compatible_with"/>
<meta itemprop="property" content="is_compatible_with"/>
<meta itemprop="property" content="merge_with"/>
</div>

# tf.compat.v1.Dimension

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>



Represents the value of one dimension in a TensorShape.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.Dimension(
    value
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

In TF2, members of a `TensorShape` object are integers. The `Dimension` class
is not part of TF2's data model.

Please refer to the [TensorShape section of the migration guide]
(https://www.tensorflow.org/guide/migrate/index#tensorshape) on common code
patterns adapting Dimension objects to a TF2 syntax.


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`value`
</td>
<td>
The value of this dimension, or None if it is unknown.
</td>
</tr>
</table>



## Methods

<h3 id="assert_is_compatible_with"><code>assert_is_compatible_with</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>assert_is_compatible_with(
    other
)
</code></pre>

Raises an exception if `other` is not compatible with this Dimension.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another Dimension.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If `self` and `other` are not compatible (see
is_compatible_with).
</td>
</tr>
</table>



<h3 id="is_compatible_with"><code>is_compatible_with</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_compatible_with(
    other
)
</code></pre>

Returns true if `other` is compatible with this Dimension.

Two known Dimensions are compatible if they have the same value.
An unknown Dimension is compatible with all other Dimensions.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another Dimension.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
True if this Dimension and `other` are compatible.
</td>
</tr>

</table>



<h3 id="merge_with"><code>merge_with</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>merge_with(
    other
)
</code></pre>

Returns a Dimension that combines the information in `self` and `other`.

Dimensions are combined as follows:

```python
tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(n))     ==
tf.compat.v1.Dimension(n)
tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(None))  ==
tf.compat.v1.Dimension(n)
tf.compat.v1.Dimension(None).merge_with(tf.compat.v1.Dimension(n))     ==
tf.compat.v1.Dimension(n)
# equivalent to tf.compat.v1.Dimension(None)
tf.compat.v1.Dimension(None).merge_with(tf.compat.v1.Dimension(None))

# raises ValueError for n != m
tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(m))
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another Dimension.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Dimension containing the combined information of `self` and
`other`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If `self` and `other` are not compatible (see
is_compatible_with).
</td>
</tr>
</table>



<h3 id="__add__"><code>__add__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    other
)
</code></pre>

Returns the sum of `self` and `other`.

Dimensions are summed as follows:

```python
tf.compat.v1.Dimension(m)    + tf.compat.v1.Dimension(n)     ==
tf.compat.v1.Dimension(m + n)
tf.compat.v1.Dimension(m)    + tf.compat.v1.Dimension(None)  # equiv. to
tf.compat.v1.Dimension(None)
tf.compat.v1.Dimension(None) + tf.compat.v1.Dimension(n)     # equiv. to
tf.compat.v1.Dimension(None)
tf.compat.v1.Dimension(None) + tf.compat.v1.Dimension(None)  # equiv. to
tf.compat.v1.Dimension(None)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another Dimension, or a value accepted by `as_dimension`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Dimension whose value is the sum of `self` and `other`.
</td>
</tr>

</table>



<h3 id="__bool__"><code>__bool__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__bool__()
</code></pre>

Equivalent to `bool(self.value)`.


<h3 id="__div__"><code>__div__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__div__(
    other
)
</code></pre>

DEPRECATED: Use `__floordiv__` via `x // y` instead.

This function exists only for backwards compatibility purposes; new code
should use `__floordiv__` via the syntax `x // y`.  Using `x // y`
communicates clearly that the result rounds down, and is forward compatible
to Python 3.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another `Dimension`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Dimension` whose value is the integer quotient of `self` and `other`.
</td>
</tr>

</table>



<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Returns true if `other` has the same known value as this Dimension.


<h3 id="__floordiv__"><code>__floordiv__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__floordiv__(
    other
)
</code></pre>

Returns the quotient of `self` and `other` rounded down.

Dimensions are divided as follows:

```python
tf.compat.v1.Dimension(m)    // tf.compat.v1.Dimension(n)     ==
tf.compat.v1.Dimension(m // n)
tf.compat.v1.Dimension(m)    // tf.compat.v1.Dimension(None)  # equiv. to
tf.compat.v1.Dimension(None)
tf.compat.v1.Dimension(None) // tf.compat.v1.Dimension(n)     # equiv. to
tf.compat.v1.Dimension(None)
tf.compat.v1.Dimension(None) // tf.compat.v1.Dimension(None)  # equiv. to
tf.compat.v1.Dimension(None)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another Dimension, or a value accepted by `as_dimension`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Dimension` whose value is the integer quotient of `self` and `other`.
</td>
</tr>

</table>



<h3 id="__ge__"><code>__ge__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ge__(
    other
)
</code></pre>

Returns True if `self` is known to be greater than or equal to `other`.

Dimensions are compared as follows:

```python
(tf.compat.v1.Dimension(m)    >= tf.compat.v1.Dimension(n))    == (m >= n)
(tf.compat.v1.Dimension(m)    >= tf.compat.v1.Dimension(None)) == None
(tf.compat.v1.Dimension(None) >= tf.compat.v1.Dimension(n))    == None
(tf.compat.v1.Dimension(None) >= tf.compat.v1.Dimension(None)) == None
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another Dimension.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The value of `self.value >= other.value` if both are known, otherwise
None.
</td>
</tr>

</table>



<h3 id="__gt__"><code>__gt__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__gt__(
    other
)
</code></pre>

Returns True if `self` is known to be greater than `other`.

Dimensions are compared as follows:

```python
(tf.compat.v1.Dimension(m)    > tf.compat.v1.Dimension(n))    == (m > n)
(tf.compat.v1.Dimension(m)    > tf.compat.v1.Dimension(None)) == None
(tf.compat.v1.Dimension(None) > tf.compat.v1.Dimension(n))    == None
(tf.compat.v1.Dimension(None) > tf.compat.v1.Dimension(None)) == None
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another Dimension.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The value of `self.value > other.value` if both are known, otherwise
None.
</td>
</tr>

</table>



<h3 id="__le__"><code>__le__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__le__(
    other
)
</code></pre>

Returns True if `self` is known to be less than or equal to `other`.

Dimensions are compared as follows:

```python
(tf.compat.v1.Dimension(m)    <= tf.compat.v1.Dimension(n))    == (m <= n)
(tf.compat.v1.Dimension(m)    <= tf.compat.v1.Dimension(None)) == None
(tf.compat.v1.Dimension(None) <= tf.compat.v1.Dimension(n))    == None
(tf.compat.v1.Dimension(None) <= tf.compat.v1.Dimension(None)) == None
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another Dimension.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The value of `self.value <= other.value` if both are known, otherwise
None.
</td>
</tr>

</table>



<h3 id="__lt__"><code>__lt__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__lt__(
    other
)
</code></pre>

Returns True if `self` is known to be less than `other`.

Dimensions are compared as follows:

```python
(tf.compat.v1.Dimension(m)    < tf.compat.v1.Dimension(n))    == (m < n)
(tf.compat.v1.Dimension(m)    < tf.compat.v1.Dimension(None)) == None
(tf.compat.v1.Dimension(None) < tf.compat.v1.Dimension(n))    == None
(tf.compat.v1.Dimension(None) < tf.compat.v1.Dimension(None)) == None
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another Dimension.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The value of `self.value < other.value` if both are known, otherwise
None.
</td>
</tr>

</table>



<h3 id="__mod__"><code>__mod__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__mod__(
    other
)
</code></pre>

Returns `self` modulo `other`.

Dimension modulo are computed as follows:

```python
tf.compat.v1.Dimension(m)    % tf.compat.v1.Dimension(n)     ==
tf.compat.v1.Dimension(m % n)
tf.compat.v1.Dimension(m)    % tf.compat.v1.Dimension(None)  # equiv. to
tf.compat.v1.Dimension(None)
tf.compat.v1.Dimension(None) % tf.compat.v1.Dimension(n)     # equiv. to
tf.compat.v1.Dimension(None)
tf.compat.v1.Dimension(None) % tf.compat.v1.Dimension(None)  # equiv. to
tf.compat.v1.Dimension(None)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another Dimension, or a value accepted by `as_dimension`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Dimension whose value is `self` modulo `other`.
</td>
</tr>

</table>



<h3 id="__mul__"><code>__mul__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__mul__(
    other
)
</code></pre>

Returns the product of `self` and `other`.

Dimensions are summed as follows:

```python
tf.compat.v1.Dimension(m)    * tf.compat.v1.Dimension(n)     ==
tf.compat.v1.Dimension(m * n)
tf.compat.v1.Dimension(m)    * tf.compat.v1.Dimension(None)  # equiv. to
tf.compat.v1.Dimension(None)
tf.compat.v1.Dimension(None) * tf.compat.v1.Dimension(n)     # equiv. to
tf.compat.v1.Dimension(None)
tf.compat.v1.Dimension(None) * tf.compat.v1.Dimension(None)  # equiv. to
tf.compat.v1.Dimension(None)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another Dimension, or a value accepted by `as_dimension`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Dimension whose value is the product of `self` and `other`.
</td>
</tr>

</table>



<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Returns true if `other` has a different known value from `self`.


<h3 id="__radd__"><code>__radd__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__radd__(
    other
)
</code></pre>

Returns the sum of `other` and `self`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another Dimension, or a value accepted by `as_dimension`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Dimension whose value is the sum of `self` and `other`.
</td>
</tr>

</table>



<h3 id="__rdiv__"><code>__rdiv__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rdiv__(
    other
)
</code></pre>

Use `__floordiv__` via `x // y` instead.

This function exists only to have a better error message. Instead of:
`TypeError: unsupported operand type(s) for /: 'int' and 'Dimension'`,
this function will explicitly call for usage of `//` instead.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another `Dimension`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>
<tr class="alt">
<td colspan="2">
TypeError.
</td>
</tr>

</table>



<h3 id="__rfloordiv__"><code>__rfloordiv__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rfloordiv__(
    other
)
</code></pre>

Returns the quotient of `other` and `self` rounded down.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another Dimension, or a value accepted by `as_dimension`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Dimension` whose value is the integer quotient of `self` and `other`.
</td>
</tr>

</table>



<h3 id="__rmod__"><code>__rmod__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rmod__(
    other
)
</code></pre>

Returns `other` modulo `self`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another Dimension, or a value accepted by `as_dimension`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Dimension whose value is `other` modulo `self`.
</td>
</tr>

</table>



<h3 id="__rmul__"><code>__rmul__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rmul__(
    other
)
</code></pre>

Returns the product of `self` and `other`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another Dimension, or a value accepted by `as_dimension`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Dimension whose value is the product of `self` and `other`.
</td>
</tr>

</table>



<h3 id="__rsub__"><code>__rsub__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rsub__(
    other
)
</code></pre>

Returns the subtraction of `self` from `other`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another Dimension, or a value accepted by `as_dimension`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Dimension whose value is the subtraction of `self` from `other`.
</td>
</tr>

</table>



<h3 id="__rtruediv__"><code>__rtruediv__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rtruediv__(
    other
)
</code></pre>

Use `__floordiv__` via `x // y` instead.

This function exists only to have a better error message. Instead of:
`TypeError: unsupported operand type(s) for /: 'int' and 'Dimension'`,
this function will explicitly call for usage of `//` instead.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another `Dimension`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>
<tr class="alt">
<td colspan="2">
TypeError.
</td>
</tr>

</table>



<h3 id="__sub__"><code>__sub__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__sub__(
    other
)
</code></pre>

Returns the subtraction of `other` from `self`.

Dimensions are subtracted as follows:

```python
tf.compat.v1.Dimension(m)    - tf.compat.v1.Dimension(n)     ==
tf.compat.v1.Dimension(m - n)
tf.compat.v1.Dimension(m)    - tf.compat.v1.Dimension(None)  # equiv. to
tf.compat.v1.Dimension(None)
tf.compat.v1.Dimension(None) - tf.compat.v1.Dimension(n)     # equiv. to
tf.compat.v1.Dimension(None)
tf.compat.v1.Dimension(None) - tf.compat.v1.Dimension(None)  # equiv. to
tf.compat.v1.Dimension(None)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another Dimension, or a value accepted by `as_dimension`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Dimension whose value is the subtraction of `other` from `self`.
</td>
</tr>

</table>



<h3 id="__truediv__"><code>__truediv__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__truediv__(
    other
)
</code></pre>

Use `__floordiv__` via `x // y` instead.

This function exists only to have a better error message. Instead of:
`TypeError: unsupported operand type(s) for /: 'Dimension' and 'int'`,
this function will explicitly call for usage of `//` instead.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another `Dimension`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>
<tr class="alt">
<td colspan="2">
TypeError.
</td>
</tr>

</table>





