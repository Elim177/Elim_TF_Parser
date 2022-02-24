description: Returns True if first argument is a typecode lower/equal in type hierarchy.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.numpy.issubdtype" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.numpy.issubdtype

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Returns True if first argument is a typecode lower/equal in type hierarchy.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.numpy.issubdtype(
    arg1, arg2
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is like the builtin :func:`issubclass`, but for `dtype`\ s.

Parameters
----------
arg1, arg2 : dtype_like
    `dtype` or object coercible to one

Returns
-------
out : bool

See Also
--------
:ref:`arrays.scalars` : Overview of the numpy type hierarchy.
issubsctype, issubclass_

Examples
--------
`issubdtype` can be used to check the type of arrays:

```
>>> ints = np.array([1, 2, 3], dtype=np.int32)
>>> np.issubdtype(ints.dtype, np.integer)
True
>>> np.issubdtype(ints.dtype, np.floating)
False
```

```
>>> floats = np.array([1, 2, 3], dtype=np.float32)
>>> np.issubdtype(floats.dtype, np.integer)
False
>>> np.issubdtype(floats.dtype, np.floating)
True
```

Similar types of different sizes are not subdtypes of each other:

```
>>> np.issubdtype(np.float64, np.float32)
False
>>> np.issubdtype(np.float32, np.float64)
False
```

but both are subtypes of `floating`:

```
>>> np.issubdtype(np.float64, np.floating)
True
>>> np.issubdtype(np.float32, np.floating)
True
```

For convenience, dtype-like objects are allowed too:

```
>>> np.issubdtype('S1', np.string_)
True
>>> np.issubdtype('i4', np.signedinteger)
True
```