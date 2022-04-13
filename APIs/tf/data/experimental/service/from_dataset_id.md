description: Creates a dataset which reads data from the tf.data service.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.service.from_dataset_id" />
<meta itemprop="path" content="Stable" />
</div>

# tf.data.experimental.service.from_dataset_id

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/data/experimental/ops/data_service_ops.py">View source</a>



Creates a dataset which reads data from the tf.data service.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.data.experimental.service.from_dataset_id`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.service.from_dataset_id(
    processing_mode, service, dataset_id, element_spec=None, job_name=None,
    consumer_index=None, num_consumers=None, max_outstanding_requests=None,
    data_transfer_protocol=None, target_workers=&#x27;AUTO&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is useful when the dataset is registered by one process, then used in
another process. When the same process is both registering and reading from
the dataset, it is simpler to use <a href="../../../../tf/data/experimental/service/distribute.md"><code>tf.data.experimental.service.distribute</code></a>
instead.

Before using `from_dataset_id`, the dataset must have been registered with the
tf.data service using <a href="../../../../tf/data/experimental/service/register_dataset.md"><code>tf.data.experimental.service.register_dataset</code></a>.
`register_dataset` returns a dataset id for the registered dataset. That is
the `dataset_id` which should be passed to `from_dataset_id`.

The `element_spec` argument indicates the <a href="../../../../tf/TypeSpec.md"><code>tf.TypeSpec</code></a>s for the elements
produced by the dataset. Currently `element_spec` must be explicitly
specified, and match the dataset registered under `dataset_id`. `element_spec`
defaults to `None` so that in the future we can support automatically
discovering the `element_spec` by querying the tf.data service.

<a href="../../../../tf/data/experimental/service/distribute.md"><code>tf.data.experimental.service.distribute</code></a> is a convenience method which
combines `register_dataset` and `from_dataset_id` into a dataset
transformation.
See the documentation for <a href="../../../../tf/data/experimental/service/distribute.md"><code>tf.data.experimental.service.distribute</code></a> for more
detail about how `from_dataset_id` works.

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
`processing_mode`
</td>
<td>
A <a href="../../../../tf/data/experimental/service/ShardingPolicy.md"><code>tf.data.experimental.service.ShardingPolicy</code></a> specifying
how to shard the dataset among tf.data workers. See
<a href="../../../../tf/data/experimental/service/ShardingPolicy.md"><code>tf.data.experimental.service.ShardingPolicy</code></a> for details. For backwards
compatibility, `processing_mode` may also be set to the strings
`"parallel_epochs"` or `"distributed_epoch"`, which are respectively
equivalent to <a href="../../../../tf/data/experimental/service/ShardingPolicy.md#OFF"><code>ShardingPolicy.OFF</code></a> and <a href="../../../../tf/data/experimental/service/ShardingPolicy.md#DYNAMIC"><code>ShardingPolicy.DYNAMIC</code></a>.
</td>
</tr><tr>
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
`dataset_id`
</td>
<td>
The id of the dataset to read from. This id is returned by
`register_dataset` when the dataset is registered with the tf.data
service.
</td>
</tr><tr>
<td>
`element_spec`
</td>
<td>
A nested structure of <a href="../../../../tf/TypeSpec.md"><code>tf.TypeSpec</code></a>s representing the type of
elements produced by the dataset. This argument is only required inside a
tf.function. Use <a href="../../../../tf/data/Dataset.md#element_spec"><code>tf.data.Dataset.element_spec</code></a> to get the element spec
for a given dataset.
</td>
</tr><tr>
<td>
`job_name`
</td>
<td>
(Optional.) The name of the job. If provided, it must be a
non-empty string. This argument makes it possible for multiple datasets to
share the same job. The default behavior is that the dataset creates
anonymous, exclusively owned jobs.
</td>
</tr><tr>
<td>
`consumer_index`
</td>
<td>
(Optional.) The index of the consumer in the range from `0`
to `num_consumers`. Must be specified alongside `num_consumers`. When
specified, consumers will read from the job in a strict round-robin order,
instead of the default first-come-first-served order.
</td>
</tr><tr>
<td>
`num_consumers`
</td>
<td>
(Optional.) The number of consumers which will consume from
the job. Must be specified alongside `consumer_index`. When specified,
consumers will read from the job in a strict round-robin order, instead of
the default first-come-first-served order. When `num_consumers` is
specified, the dataset must have infinite cardinality to prevent a
producer from running out of data early and causing consumers to go out of
sync.
</td>
</tr><tr>
<td>
`max_outstanding_requests`
</td>
<td>
(Optional.) A limit on how many elements may be
requested at the same time. You can use this option to control the amount
of memory used, since `distribute` won't use more than `element_size` *
`max_outstanding_requests` of memory.
</td>
</tr><tr>
<td>
`data_transfer_protocol`
</td>
<td>
(Optional.) The protocol to use for transferring
data with the tf.data service. By default, data is transferred using gRPC.
</td>
</tr><tr>
<td>
`target_workers`
</td>
<td>
(Optional.) Which workers to read from. If `"AUTO"`, tf.data
runtime decides which workers to read from. If `"ANY"`, reads from any
tf.data service workers. If `"LOCAL"`, only reads from local in-processs
tf.data service workers. `"AUTO"` works well for most cases, while users
can specify other targets. For example, `"LOCAL"` helps avoid RPCs and
data copy if every TF worker colocates with a tf.data service worker.
Consumers of a shared job must use the same `target_workers`. Defaults to
`"AUTO"`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> which reads from the tf.data service.
</td>
</tr>

</table>

