description: Returns an Op that initializes all tables of the default graph.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.tables_initializer" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.tables_initializer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/lookup_ops.py">View source</a>



Returns an Op that initializes all tables of the default graph.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.initializers.tables_initializer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.tables_initializer(
    name=&#x27;init_all_tables&#x27;
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

<a href="../../../tf/compat/v1/tables_initializer.md"><code>tf.compat.v1.tables_initializer</code></a> is no longer needed with eager execution and
<a href="../../../tf/function.md"><code>tf.function</code></a>. In TF2, when creating an initializable table like a
<a href="../../../tf/lookup/StaticHashTable.md"><code>tf.lookup.StaticHashTable</code></a>, the table will automatically be initialized on
creation.

#### Before & After Usage Example

Before:

```
>>> with tf.compat.v1.Session():
...   init = tf.compat.v1.lookup.KeyValueTensorInitializer(['a', 'b'], [1, 2])
...   table = tf.compat.v1.lookup.StaticHashTable(init, default_value=-1)
...   tf.compat.v1.tables_initializer().run()
...   result = table.lookup(tf.constant(['a', 'c'])).eval()
>>> result
array([ 1, -1], dtype=int32)
```

After:

```
>>> init = tf.lookup.KeyValueTensorInitializer(['a', 'b'], [1, 2])
>>> table = tf.lookup.StaticHashTable(init, default_value=-1)
>>> table.lookup(tf.constant(['a', 'c'])).numpy()
array([ 1, -1], dtype=int32)
```



 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`
</td>
<td>
Optional name for the initialization op.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An Op that initializes all tables.  Note that if there are
not tables the returned Op is a NoOp.
</td>
</tr>

</table>


