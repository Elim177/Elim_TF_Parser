description: Creates a uninitialized anonymous hash table.
robots: noindex

# tf.raw_ops.AnonymousHashTable

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Creates a uninitialized anonymous hash table.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.AnonymousHashTable`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.AnonymousHashTable(
    key_dtype, value_dtype, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This op creates a new anonymous hash table (as a resource) everytime
it is executed, with the specified dtype of its keys and values,
returning the resource handle.  Before using the table you will have
to initialize it.  After initialization the table will be
immutable. The table is anonymous in the sense that it can only be
accessed by the returned resource handle (e.g. it cannot be looked up
by a name in a resource manager). The table will be automatically
deleted when all resource handles pointing to it are gone.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`key_dtype`
</td>
<td>
A <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a>. Type of the table keys.
</td>
</tr><tr>
<td>
`value_dtype`
</td>
<td>
A <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a>. Type of the table values.
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
A `Tensor` of type `resource`.
</td>
</tr>

</table>

