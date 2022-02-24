description: Applies func to each entry in structure and returns a new structure.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nest.map_structure" />
<meta itemprop="path" content="Stable" />
</div>

# tf.nest.map_structure

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/util/nest.py">View source</a>



Applies `func` to each entry in `structure` and returns a new structure.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.nest.map_structure`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.nest.map_structure(
    func, *structure, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Applies `func(x[0], x[1], ...)` where x[i] is an entry in
`structure[i]`.  All structures in `structure` must have the same arity,
and the return value will contain results with the same structure layout.

#### Examples:



* A single Python dict:

```
>>> a = {"hello": 24, "world": 76}
>>> tf.nest.map_structure(lambda p: p * 2, a)
{'hello': 48, 'world': 152}
```

* Multiple Python dictionaries:

```
>>> d1 = {"hello": 24, "world": 76}
>>> d2 = {"hello": 36, "world": 14}
>>> tf.nest.map_structure(lambda p1, p2: p1 + p2, d1, d2)
{'hello': 60, 'world': 90}
```

* A single Python list:

```
>>> a = [24, 76, "ab"]
>>> tf.nest.map_structure(lambda p: p * 2, a)
[48, 152, 'abab']
```

* Scalars:

```
>>> tf.nest.map_structure(lambda x, y: x + y, 3, 4)
7
```

* Empty structures:

```
>>> tf.nest.map_structure(lambda x: x + 1, ())
()
```

* Check the types of iterables:

```
>>> s1 = (((1, 2), 3), 4, (5, 6))
>>> s1_list = [[[1, 2], 3], 4, [5, 6]]
>>> tf.nest.map_structure(lambda x, y: None, s1, s1_list)
Traceback (most recent call last):
...
TypeError: The two structures don't have the same nested structure
```

* Type check is set to False:

```
>>> s1 = (((1, 2), 3), 4, (5, 6))
>>> s1_list = [[[1, 2], 3], 4, [5, 6]]
>>> tf.nest.map_structure(lambda x, y: None, s1, s1_list, check_types=False)
(((None, None), None), None, (None, None))
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`func`
</td>
<td>
A callable that accepts as many arguments as there are structures.
</td>
</tr><tr>
<td>
`*structure`
</td>
<td>
scalar, or tuple or dict or list of constructed scalars and/or
other tuples/lists, or scalars.  Note: numpy arrays are considered as
scalars.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Valid keyword args are:

* `check_types`: If set to `True` (default) the types of
  iterables within the structures have to be same (e.g.
  `map_structure(func, [1], (1,))` raises a `TypeError`
  exception). To allow this set this argument to `False`.
  Note that namedtuples with identical name and fields are always
  considered to have the same shallow structure.
* `expand_composites`: If set to `True`, then composite tensors such
  as <a href="../../tf/sparse/SparseTensor.md"><code>tf.sparse.SparseTensor</code></a> and <a href="../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a> are expanded into
  their component tensors.  If `False` (the default), then composite
  tensors are not expanded.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A new structure with the same arity as `structure`, whose values correspond
to `func(x[0], x[1], ...)` where `x[i]` is a value in the corresponding
location in `structure[i]`. If there are different sequence types and
`check_types` is `False` the sequence types of the first structure will be
used.
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
If `func` is not callable or if the structures do not match
each other by depth tree.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If no structure is provided or if the structures do not match
each other by type.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If wrong keyword arguments are provided.
</td>
</tr>
</table>

