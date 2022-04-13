description: Holds a Tensor which a tf.function can capture.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.saved_model.experimental.TrackableResource" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.saved_model.experimental.TrackableResource

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/training/tracking/tracking.py">View source</a>



Holds a Tensor which a tf.function can capture.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.saved_model.experimental.TrackableResource`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.saved_model.experimental.TrackableResource(
    device=&#x27;&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

A TrackableResource is most useful for stateful Tensors that require
initialization, such as <a href="../../../tf/lookup/StaticHashTable.md"><code>tf.lookup.StaticHashTable</code></a>. `TrackableResource`s
are discovered by traversing the graph of object attributes, e.g. during
<a href="../../../tf/saved_model/save.md"><code>tf.saved_model.save</code></a>.

A TrackableResource has three methods to override:

* `_create_resource` should create the resource tensor handle.
* `_initialize` should initialize the resource held at `self.resource_handle`.
* `_destroy_resource` is called upon a `TrackableResource`'s destruction
  and should decrement the resource's ref count. For most resources, this
  should be done with a call to <a href="../../../tf/raw_ops/DestroyResourceOp.md"><code>tf.raw_ops.DestroyResourceOp</code></a>.

#### Example usage:



```
>>> class DemoResource(tf.saved_model.experimental.TrackableResource):
...   def __init__(self):
...     super().__init__()
...     self._initialize()
...   def _create_resource(self):
...     return tf.raw_ops.VarHandleOp(dtype=tf.float32, shape=[2])
...   def _initialize(self):
...     tf.raw_ops.AssignVariableOp(
...         resource=self.resource_handle, value=tf.ones([2]))
...   def _destroy_resource(self):
...     tf.raw_ops.DestroyResourceOp(resource=self.resource_handle)
>>> class DemoModule(tf.Module):
...   def __init__(self):
...     self.resource = DemoResource()
...   def increment(self, tensor):
...     return tensor + tf.raw_ops.ReadVariableOp(
...         resource=self.resource.resource_handle, dtype=tf.float32)
>>> demo = DemoModule()
>>> demo.increment([5, 1])
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([6., 2.], dtype=float32)>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`device`
</td>
<td>
A string indicating a required placement for this resource,
e.g. "CPU" if this resource must be created on a CPU device. A blank
device allows the user to place resource creation, so generally this
should be blank unless the resource only makes sense on one device.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`resource_handle`
</td>
<td>
Returns the resource handle associated with this Resource.
</td>
</tr>
</table>



