description: A protobuf containing the graph of operations.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.GraphDef" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.GraphDef

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/core/framework/graph.proto">View source</a>



A protobuf containing the graph of operations.



 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

This API is not available in TensorFlow 2.x.

You should not need to use `GraphDef`s directly in TF2. To load `GraphDef`s in
TF2, use SavedModel. The SavedModel contains the `GraphDef`.

Before:

```python
with tf.io.gfile.GFile('/tmp/graph.pb', 'rb') as f:
  graph_def = tf.compat.v1.GraphDef()
  graph_def.ParseFromString(f.read())
```

After:

```python
tf.saved_model.load('/tmp/saved_model')
```

If you would like to create a `GraphDef` in TF2, use <a href="../../../tf/function.md"><code>tf.function</code></a> and
`get_concrete_function`.

```
>>> @tf.function
>>> def f(x):
>>>   return x
>>>
>>> graph_def = f.get_concrete_function(1.).graph.as_graph_def()
>>> print(graph_def)
```



 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`library`
</td>
<td>
`FunctionDefLibrary library`
</td>
</tr><tr>
<td>
`node`
</td>
<td>
`repeated NodeDef node`
</td>
</tr><tr>
<td>
`version`
</td>
<td>
`int32 version`
</td>
</tr><tr>
<td>
`versions`
</td>
<td>
`VersionDef versions`
</td>
</tr>
</table>



