description: Returns a flat list from a given nested structure.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nest.flatten" />
<meta itemprop="path" content="Stable" />
</div>

# tf.nest.flatten

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/util/nest.py">View source</a>



Returns a flat list from a given nested structure.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.nest.flatten`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.nest.flatten(
    structure, expand_composites=(False)
)
</code></pre>



<!-- Placeholder for "Used in" -->

If nest is not a structure , tuple (or a namedtuple), dict, or an attrs class,
then returns a single-element list:
  [nest].

This is the inverse of the <a href="../../tf/nest/pack_sequence_as.md"><code>nest.pack_sequence_as</code></a> method that takes in a
flattened list and re-packs it into the nested structure.

In the case of dict instances, the sequence consists of the values, sorted by
key to ensure deterministic behavior. This is true also for OrderedDict
instances: their sequence order is ignored, the sorting order of keys is used
instead. The same convention is followed in <a href="../../tf/nest/pack_sequence_as.md"><code>nest.pack_sequence_as</code></a>. This
correctly repacks dicts and OrderedDicts after they have been flattened, and
also allows flattening an OrderedDict and then repacking it back using a
corresponding plain dict, or vice-versa. Dictionaries with non-sortable keys
cannot be flattened.

Users must not modify any collections used in nest while this function is
running.

#### Examples:



1. Python dict (ordered by key):

  ```
  >>> dict = { "key3": "value3", "key1": "value1", "key2": "value2" }
  >>> tf.nest.flatten(dict)
  ['value1', 'value2', 'value3']
  ```

2. For a nested python tuple:

  ```
  >>> tuple = ((1.0, 2.0), (3.0, 4.0, 5.0), 6.0)
  >>> tf.nest.flatten(tuple)
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  ```

3. For a nested dictionary of dictionaries:

  ```
  >>> dict = { "key3": {"c": (1.0, 2.0), "a": (3.0)},
  ... "key1": {"m": "val1", "g": "val2"} }
  >>> tf.nest.flatten(dict)
  ['val2', 'val1', 3.0, 1.0, 2.0]
  ```

4. Numpy array (will not flatten):

  ```
  >>> array = np.array([[1, 2], [3, 4]])
  >>> tf.nest.flatten(array)
      [array([[1, 2],
              [3, 4]])]
  ```

5. <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a> (will not flatten):

  ```
  >>> tensor = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
  >>> tf.nest.flatten(tensor)
      [<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
        array([[1., 2., 3.],
               [4., 5., 6.],
               [7., 8., 9.]], dtype=float32)>]
  ```

6. <a href="../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a>: This is a composite tensor thats representation consists
of a flattened list of 'values' and a list of 'row_splits' which indicate how
to chop up the flattened list into different rows. For more details on
<a href="../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a>, please visit
https://www.tensorflow.org/api_docs/python/tf/RaggedTensor.

with `expand_composites=False`, we just return the RaggedTensor as is.

  ```
  >>> tensor = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2]])
  >>> tf.nest.flatten(tensor, expand_composites=False)
  [<tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2]]>]
  ```

with `expand_composites=True`, we return the component Tensors that make up
the RaggedTensor representation (the values and row_splits tensors)

  ```
  >>> tensor = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2]])
  >>> tf.nest.flatten(tensor, expand_composites=True)
  [<tf.Tensor: shape=(7,), dtype=int32, numpy=array([3, 1, 4, 1, 5, 9, 2],
                                                    dtype=int32)>,
   <tf.Tensor: shape=(4,), dtype=int64, numpy=array([0, 4, 4, 7])>]
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
an arbitrarily nested structure. Note, numpy arrays are
considered atoms and are not flattened.
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
<tr class="alt">
<td colspan="2">
A Python list, the flattened version of the input.
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
The nest is or contains a dict with non-sortable keys.
</td>
</tr>
</table>

