description: Represents the type of the elements in a Tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.dtypes.DType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="is_compatible_with"/>
</div>

# tf.dtypes.DType

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/dtypes.py">View source</a>



Represents the type of the elements in a `Tensor`.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.DType`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.DType`, `tf.compat.v1.dtypes.DType`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.dtypes.DType()
</code></pre>



<!-- Placeholder for "Used in" -->

`DType`s are used to specify the output data type for operations which
require it, or to inspect the data type of existing `Tensor`s.

#### Examples:



```
>>> tf.constant(1, dtype=tf.int64)
<tf.Tensor: shape=(), dtype=int64, numpy=1>
>>> tf.constant(1.0).dtype
tf.float32
```

See <a href="../../tf/dtypes.md"><code>tf.dtypes</code></a> for a complete list of `DType`s defined.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`as_datatype_enum`
</td>
<td>
Returns a `types_pb2.DataType` enum value based on this data type.
</td>
</tr><tr>
<td>
`as_numpy_dtype`
</td>
<td>
Returns a Python `type` object based on this `DType`.
</td>
</tr><tr>
<td>
`base_dtype`
</td>
<td>
Returns a non-reference `DType` based on this `DType`.
</td>
</tr><tr>
<td>
`is_bool`
</td>
<td>
Returns whether this is a boolean data type.
</td>
</tr><tr>
<td>
`is_complex`
</td>
<td>
Returns whether this is a complex floating point type.
</td>
</tr><tr>
<td>
`is_floating`
</td>
<td>
Returns whether this is a (non-quantized, real) floating point type.
</td>
</tr><tr>
<td>
`is_integer`
</td>
<td>
Returns whether this is a (non-quantized) integer type.
</td>
</tr><tr>
<td>
`is_numpy_compatible`
</td>
<td>
Returns whether this data type has a compatible NumPy data type.
</td>
</tr><tr>
<td>
`is_quantized`
</td>
<td>
Returns whether this is a quantized data type.
</td>
</tr><tr>
<td>
`is_unsigned`
</td>
<td>
Returns whether this type is unsigned.

Non-numeric, unordered, and quantized types are not considered unsigned, and
this function returns `False`.
</td>
</tr><tr>
<td>
`limits`
</td>
<td>
Return intensity limits, i.e.

(min, max) tuple, of the dtype.
Args:
  clip_negative : bool, optional If True, clip the negative range (i.e.
    return 0 for min intensity) even if the image dtype allows negative
    values. Returns
  min, max : tuple Lower and upper intensity limits.
</td>
</tr><tr>
<td>
`max`
</td>
<td>
Returns the maximum representable value in this data type.
</td>
</tr><tr>
<td>
`min`
</td>
<td>
Returns the minimum representable value in this data type.
</td>
</tr><tr>
<td>
`name`
</td>
<td>

</td>
</tr><tr>
<td>
`real_dtype`
</td>
<td>
Returns the `DType` corresponding to this `DType`'s real part.
</td>
</tr><tr>
<td>
`size`
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="is_compatible_with"><code>is_compatible_with</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/dtypes.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_compatible_with(
    other
)
</code></pre>

Returns True if the `other` DType will be converted to this DType.

The conversion rules are as follows:

```python
DType(T)       .is_compatible_with(DType(T))        == True
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
A `DType` (or object that may be converted to a `DType`).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
True if a Tensor of the `other` `DType` will be implicitly converted to
this `DType`.
</td>
</tr>

</table>



<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/dtypes.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Returns True iff this DType refers to the same type as `other`.


<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/dtypes.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Returns True iff self != other.




