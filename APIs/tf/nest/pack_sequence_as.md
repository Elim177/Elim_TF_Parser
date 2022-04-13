description: Returns a given flattened sequence packed into a given structure.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nest.pack_sequence_as" />
<meta itemprop="path" content="Stable" />
</div>

# tf.nest.pack_sequence_as

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/util/nest.py">View source</a>



Returns a given flattened sequence packed into a given structure.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.nest.pack_sequence_as`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.nest.pack_sequence_as(
    structure, flat_sequence, expand_composites=(False)
)
</code></pre>



<!-- Placeholder for "Used in" -->

If `structure` is a scalar, `flat_sequence` must be a single-element list;
in this case the return value is `flat_sequence[0]`.

If `structure` is or contains a dict instance, the keys will be sorted to
pack the flat sequence in deterministic order. This is true also for
`OrderedDict` instances: their sequence order is ignored, the sorting order of
keys is used instead. The same convention is followed in `flatten`.
This correctly repacks dicts and `OrderedDict`s after they have been
flattened, and also allows flattening an `OrderedDict` and then repacking it
back using a corresponding plain dict, or vice-versa.
Dictionaries with non-sortable keys cannot be flattened.

#### Examples:



1. Python dict:

  ```
  >>> structure = { "key3": "", "key1": "", "key2": "" }
  >>> flat_sequence = ["value1", "value2", "value3"]
  >>> tf.nest.pack_sequence_as(structure, flat_sequence)
  {'key3': 'value3', 'key1': 'value1', 'key2': 'value2'}
  ```

2. For a nested python tuple:

  ```
  >>> structure = (('a','b'), ('c','d','e'), 'f')
  >>> flat_sequence = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  >>> tf.nest.pack_sequence_as(structure, flat_sequence)
  ((1.0, 2.0), (3.0, 4.0, 5.0), 6.0)
  ```

3. For a nested dictionary of dictionaries:

  ```
  >>> structure = { "key3": {"c": ('alpha', 'beta'), "a": ('gamma')},
  ...               "key1": {"e": "val1", "d": "val2"} }
  >>> flat_sequence = ['val2', 'val1', 3.0, 1.0, 2.0]
  >>> tf.nest.pack_sequence_as(structure, flat_sequence)
  {'key3': {'c': (1.0, 2.0), 'a': 3.0}, 'key1': {'e': 'val1', 'd': 'val2'}}
  ```

4. Numpy array (considered a scalar):

  ```
  >>> structure = ['a']
  >>> flat_sequence = [np.array([[1, 2], [3, 4]])]
  >>> tf.nest.pack_sequence_as(structure, flat_sequence)
  [array([[1, 2],
         [3, 4]])]
  ```

5. tf.Tensor (considered a scalar):

  ```
  >>> structure = ['a']
  >>> flat_sequence = [tf.constant([[1., 2., 3.], [4., 5., 6.]])]
  >>> tf.nest.pack_sequence_as(structure, flat_sequence)
  [<tf.Tensor: shape=(2, 3), dtype=float32,
   numpy= array([[1., 2., 3.], [4., 5., 6.]], dtype=float32)>]
  ```

6. <a href="../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a>: This is a composite tensor thats representation consists
of a flattened list of 'values' and a list of 'row_splits' which indicate how
to chop up the flattened list into different rows. For more details on
<a href="../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a>, please visit
https://www.tensorflow.org/api_docs/python/tf/RaggedTensor.

With `expand_composites=False`, we treat RaggedTensor as a scalar.

  ```
  >>> structure = { "foo": tf.ragged.constant([[1, 2], [3]]),
  ...               "bar": tf.constant([[5]]) }
  >>> flat_sequence = [ "one", "two" ]
  >>> tf.nest.pack_sequence_as(structure, flat_sequence,
  ... expand_composites=False)
  {'foo': 'two', 'bar': 'one'}
  ```

With `expand_composites=True`, we expect that the flattened input contains
the tensors making up the ragged tensor i.e. the values and row_splits
tensors.

  ```
  >>> structure = { "foo": tf.ragged.constant([[1., 2.], [3.]]),
  ...               "bar": tf.constant([[5.]]) }
  >>> tensors = tf.nest.flatten(structure, expand_composites=True)
  >>> print(tensors)
  [<tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[5.]],
   dtype=float32)>,
   <tf.Tensor: shape=(3,), dtype=float32, numpy=array([1., 2., 3.],
   dtype=float32)>,
   <tf.Tensor: shape=(3,), dtype=int64, numpy=array([0, 2, 3])>]
  >>> verified_tensors = [tf.debugging.check_numerics(t, 'invalid tensor: ')
  ...                     if t.dtype==tf.float32 else t
  ...                     for t in tensors]
  >>> tf.nest.pack_sequence_as(structure, verified_tensors,
  ...                          expand_composites=True)
  {'foo': <tf.RaggedTensor [[1.0, 2.0], [3.0]]>,
   'bar': <tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[5.]],
   dtype=float32)>}
  ```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`structure`
</td>
<td>
Nested structure, whose structure is given by nested lists,
tuples, and dicts. Note: numpy arrays and strings are considered
scalars.
</td>
</tr><tr>
<td>
`flat_sequence`
</td>
<td>
flat sequence to pack.
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
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`packed`
</td>
<td>
`flat_sequence` converted to have the same recursive structure as
`structure`.
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
If `flat_sequence` and `structure` have different
element counts.
</td>
</tr><tr>
<td>
`TypeError`
</td>
<td>
`structure` is or contains a dict with non-sortable keys.
</td>
</tr>
</table>

