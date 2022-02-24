description: Asserts that two structures are nested in the same way.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nest.assert_same_structure" />
<meta itemprop="path" content="Stable" />
</div>

# tf.nest.assert_same_structure

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/util/nest.py">View source</a>



Asserts that two structures are nested in the same way.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.nest.assert_same_structure`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.nest.assert_same_structure(
    nest1, nest2, check_types=(True), expand_composites=(False)
)
</code></pre>



<!-- Placeholder for "Used in" -->

Note the method does not check the types of data inside the structures.

#### Examples:



* These scalar vs. scalar comparisons will pass:

  ```
  >>> tf.nest.assert_same_structure(1.5, tf.Variable(1, tf.uint32))
  >>> tf.nest.assert_same_structure("abc", np.array([1, 2]))
  ```

* These sequence vs. sequence comparisons will pass:

  ```
  >>> structure1 = (((1, 2), 3), 4, (5, 6))
  >>> structure2 = ((("foo1", "foo2"), "foo3"), "foo4", ("foo5", "foo6"))
  >>> structure3 = [(("a", "b"), "c"), "d", ["e", "f"]]
  >>> tf.nest.assert_same_structure(structure1, structure2)
  >>> tf.nest.assert_same_structure(structure1, structure3, check_types=False)
  ```

  ```
  >>> import collections
  >>> tf.nest.assert_same_structure(
  ...     collections.namedtuple("bar", "a b")(1, 2),
  ...     collections.namedtuple("foo", "a b")(2, 3),
  ...     check_types=False)
  ```

  ```
  >>> tf.nest.assert_same_structure(
  ...     collections.namedtuple("bar", "a b")(1, 2),
  ...     { "a": 1, "b": 2 },
  ...     check_types=False)
  ```

  ```
  >>> tf.nest.assert_same_structure(
  ...     { "a": 1, "b": 2, "c": 3 },
  ...     { "c": 6, "b": 5, "a": 4 })
  ```

  ```
  >>> ragged_tensor1 = tf.RaggedTensor.from_row_splits(
  ...       values=[3, 1, 4, 1, 5, 9, 2, 6],
  ...       row_splits=[0, 4, 4, 7, 8, 8])
  >>> ragged_tensor2 = tf.RaggedTensor.from_row_splits(
  ...       values=[3, 1, 4],
  ...       row_splits=[0, 3])
  >>> tf.nest.assert_same_structure(
  ...       ragged_tensor1,
  ...       ragged_tensor2,
  ...       expand_composites=True)
  ```

* These examples will raise exceptions:

  ```
  >>> tf.nest.assert_same_structure([0, 1], np.array([0, 1]))
  Traceback (most recent call last):
  ...
  ValueError: The two structures don't have the same nested structure
  ```

  ```
  >>> tf.nest.assert_same_structure(
  ...       collections.namedtuple('bar', 'a b')(1, 2),
  ...       collections.namedtuple('foo', 'a b')(2, 3))
  Traceback (most recent call last):
  ...
  TypeError: The two structures don't have the same nested structure
  ```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`nest1`
</td>
<td>
an arbitrarily nested structure.
</td>
</tr><tr>
<td>
`nest2`
</td>
<td>
an arbitrarily nested structure.
</td>
</tr><tr>
<td>
`check_types`
</td>
<td>
if `True` (default) types of sequences are checked as well,
including the keys of dictionaries. If set to `False`, for example a
list and a tuple of objects will look the same if they have the same
size. Note that namedtuples with identical name and fields are always
considered to have the same shallow structure. Two types will also be
considered the same if they are both list subtypes (which allows "list"
and "_ListWrapper" from trackable dependency tracking to compare
equal).
</td>
</tr><tr>
<td>
`expand_composites`
</td>
<td>
If true, then composite tensors such as
<a href="../../tf/sparse/SparseTensor.md"><code>tf.sparse.SparseTensor</code></a> and <a href="../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a> are expanded into their
component tensors.
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
If the two structures do not have the same number of elements or
if the two structures are not nested in the same way.
</td>
</tr><tr>
<td>
`TypeError`
</td>
<td>
If the two structures differ in the type of sequence in any of
their substructures. Only possible if `check_types` is `True`.
</td>
</tr>
</table>

