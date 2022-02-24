description: A generic mutable hash table implementation.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.lookup.experimental.MutableHashTable" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="export"/>
<meta itemprop="property" content="insert"/>
<meta itemprop="property" content="lookup"/>
<meta itemprop="property" content="remove"/>
<meta itemprop="property" content="size"/>
</div>

# tf.lookup.experimental.MutableHashTable

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/lookup_ops.py">View source</a>



A generic mutable hash table implementation.

Inherits From: [`TrackableResource`](../../../tf/saved_model/experimental/TrackableResource.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.lookup.experimental.MutableHashTable`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.lookup.experimental.MutableHashTable(
    key_dtype, value_dtype, default_value, name=&#x27;MutableHashTable&#x27;,
    checkpoint=(True)
)
</code></pre>



<!-- Placeholder for "Used in" -->

Data can be inserted by calling the `insert` method and removed by calling the
`remove` method. It does not support initialization via the init method.

`MutableHashTable` requires additional memory during checkpointing and restore
operations to create temporary key and value tensors.

#### Example usage:



```
>>> table = tf.lookup.experimental.MutableHashTable(key_dtype=tf.string,
...                                                 value_dtype=tf.int64,
...                                                 default_value=-1)
>>> keys_tensor = tf.constant(['a', 'b', 'c'])
>>> vals_tensor = tf.constant([7, 8, 9], dtype=tf.int64)
>>> input_tensor = tf.constant(['a', 'f'])
>>> table.insert(keys_tensor, vals_tensor)
>>> table.lookup(input_tensor).numpy()
array([ 7, -1])
>>> table.remove(tf.constant(['c']))
>>> table.lookup(keys_tensor).numpy()
array([ 7, 8, -1])
>>> sorted(table.export()[0].numpy())
[b'a', b'b']
>>> sorted(table.export()[1].numpy())
[7, 8]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`key_dtype`
</td>
<td>
the type of the key tensors.
</td>
</tr><tr>
<td>
`value_dtype`
</td>
<td>
the type of the value tensors.
</td>
</tr><tr>
<td>
`default_value`
</td>
<td>
The value to use if a key is missing in the table.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr><tr>
<td>
`checkpoint`
</td>
<td>
if True, the contents of the table are saved to and restored
from checkpoints. If `shared_name` is empty for a checkpointed table, it
is shared using the table node name.
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
If checkpoint is True and no name was specified.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`key_dtype`
</td>
<td>
The table key dtype.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
The name of the table.
</td>
</tr><tr>
<td>
`resource_handle`
</td>
<td>
Returns the resource handle associated with this Resource.
</td>
</tr><tr>
<td>
`value_dtype`
</td>
<td>
The table value dtype.
</td>
</tr>
</table>



## Methods

<h3 id="export"><code>export</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/ops/lookup_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>export(
    name=None
)
</code></pre>

Returns tensors of all keys and values in the table.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
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
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A pair of tensors with the first tensor containing all keys and the
second tensors containing all values in the table.
</td>
</tr>

</table>



<h3 id="insert"><code>insert</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/ops/lookup_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>insert(
    keys, values, name=None
)
</code></pre>

Associates `keys` with `values`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`keys`
</td>
<td>
Keys to insert. Can be a tensor of any shape. Must match the table's
key type.
</td>
</tr><tr>
<td>
`values`
</td>
<td>
Values to be associated with keys. Must be a tensor of the same
shape as `keys` and match the table's value type.
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
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The created Operation.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
when `keys` or `values` doesn't match the table data
types.
</td>
</tr>
</table>



<h3 id="lookup"><code>lookup</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/ops/lookup_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>lookup(
    keys, dynamic_default_values=None, name=None
)
</code></pre>

Looks up `keys` in a table, outputs the corresponding values.

The `default_value` is used for keys not present in the table.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`keys`
</td>
<td>
Keys to look up. Can be a tensor of any shape. Must match the
table's key_dtype.
</td>
</tr><tr>
<td>
`dynamic_default_values`
</td>
<td>
The values to use if a key is missing in the
table. If None (by default), the `table.default_value` will be used.
Shape of `dynamic_default_values` must be same with
`table.default_value` or the lookup result tensor.
In the latter case, each key will have a different default value.

For example:

  ```python
  keys = [0, 1, 3]
  dynamic_default_values = [[1, 3, 4], [2, 3, 9], [8, 3, 0]]

  # The key '0' will use [1, 3, 4] as default value.
  # The key '1' will use [2, 3, 9] as default value.
  # The key '3' will use [8, 3, 0] as default value.
  ```
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
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tensor containing the values in the same shape as `keys` using the
table's value type.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
when `keys` do not match the table data types.
</td>
</tr>
</table>



<h3 id="remove"><code>remove</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/ops/lookup_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>remove(
    keys, name=None
)
</code></pre>

Removes `keys` and its associated values from the table.

If a key is not present in the table, it is silently ignored.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`keys`
</td>
<td>
Keys to remove. Can be a tensor of any shape. Must match the table's
key type.
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
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The created Operation.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
when `keys` do not match the table data types.
</td>
</tr>
</table>



<h3 id="size"><code>size</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/ops/lookup_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>size(
    name=None
)
</code></pre>

Compute the number of elements in this table.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
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
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A scalar tensor containing the number of elements in this table.
</td>
</tr>

</table>



<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/ops/lookup_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    keys
)
</code></pre>

Looks up `keys` in a table, outputs the corresponding values.




