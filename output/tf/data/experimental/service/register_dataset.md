description: Registers a dataset with the tf.data service.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.service.register_dataset" />
<meta itemprop="path" content="Stable" />
</div>

# tf.data.experimental.service.register_dataset

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/data/experimental/ops/data_service_ops.py">View source</a>



Registers a dataset with the tf.data service.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.data.experimental.service.register_dataset`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.service.register_dataset(
    service, dataset, compression=&#x27;AUTO&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

`register_dataset` registers a dataset with the tf.data service so that
datasets can be created later with
<a href="../../../../tf/data/experimental/service/from_dataset_id.md"><code>tf.data.experimental.service.from_dataset_id</code></a>. This is useful when the
dataset
is registered by one process, then used in another process. When the same
process is both registering and reading from the dataset, it is simpler to use
<a href="../../../../tf/data/experimental/service/distribute.md"><code>tf.data.experimental.service.distribute</code></a> instead.

If the dataset is already registered with the tf.data service,
`register_dataset` returns the already-registered dataset's id.

```
>>> dispatcher = tf.data.experimental.service.DispatchServer()
>>> dispatcher_address = dispatcher.target.split("://")[1]
>>> worker = tf.data.experimental.service.WorkerServer(
...     tf.data.experimental.service.WorkerConfig(
...         dispatcher_address=dispatcher_address))
>>> dataset = tf.data.Dataset.range(10)
>>> dataset_id = tf.data.experimental.service.register_dataset(
...     dispatcher.target, dataset)
>>> dataset = tf.data.experimental.service.from_dataset_id(
...     processing_mode="parallel_epochs",
...     service=dispatcher.target,
...     dataset_id=dataset_id,
...     element_spec=dataset.element_spec)
>>> print(list(dataset.as_numpy_iterator()))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`service`
</td>
<td>
A string or a tuple indicating how to connect to the tf.data
service. If it's a string, it should be in the format
`[<protocol>://]<address>`, where `<address>` identifies the dispatcher
  address and `<protocol>` can optionally be used to override the default
  protocol to use. If it's a tuple, it should be (protocol, address).
</td>
</tr><tr>
<td>
`dataset`
</td>
<td>
A <a href="../../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> to register with the tf.data service.
</td>
</tr><tr>
<td>
`compression`
</td>
<td>
(Optional.) How to compress the dataset's elements before
transferring them over the network. "AUTO" leaves the decision of how to
compress up to the tf.data service runtime. `None` indicates not to
compress.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A scalar int64 tensor of the registered dataset's id.
</td>
</tr>

</table>

