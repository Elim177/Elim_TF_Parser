description: Specifies how to shard data among tf.data service workers.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.service.ShardingPolicy" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="DATA"/>
<meta itemprop="property" content="DYNAMIC"/>
<meta itemprop="property" content="FILE"/>
<meta itemprop="property" content="FILE_OR_DATA"/>
<meta itemprop="property" content="HINT"/>
<meta itemprop="property" content="OFF"/>
</div>

# tf.data.experimental.service.ShardingPolicy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/data/experimental/ops/data_service_ops.py">View source</a>



Specifies how to shard data among tf.data service workers.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.data.experimental.service.ShardingPolicy`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

OFF: No sharding will be performed. Each worker produces the entire dataset
without any sharding. With this mode, the best practice is to shuffle the
dataset nondeterministically so that workers process the dataset in different
orders. If workers are restarted or join the cluster mid-job, they will begin
processing the dataset from the beginning.

DYNAMIC: The input dataset is dynamically split among workers at runtime. Each
worker gets the next split when it reads data from the dispatcher. Data is
produced non-deterministically in this mode. Dynamic sharding works well with
varying-sized tf.data service clusters, e.g., when you need to auto-scale your
workers. Dynamic sharding provides at-most once visitation guarantees. No
examples will be repeated, but some may be missed if a tf.data service worker
gets restarted while processing a file.

The following are static sharding policies. The semantics are similar to
<a href="../../../../tf/data/experimental/AutoShardPolicy.md"><code>tf.data.experimental.AutoShardPolicy</code></a>. These policies require:
* The tf.data service cluster is configured with a fixed list of workers
  in DispatcherConfig.
* Each client only reads from the local tf.data service worker.

If a worker is restarted while performing static sharding, the worker will
begin processing its shard again from the beginning.

FILE: Shards by input files (i.e. each worker will get a fixed set of files to
process). When this option is selected, make sure that there is at least as
many files as workers. If there are fewer input files than workers, a runtime
error will be raised.

DATA: Shards by elements produced by the dataset. Each worker will process the
whole dataset and discard the portion that is not for itself. Note that for
this mode to correctly partition the dataset elements, the dataset needs to
produce elements in a deterministic order.

FILE_OR_DATA: Attempts FILE-based sharding, falling back to DATA-based
sharding on failure.

HINT: Looks for the presence of `shard(SHARD_HINT, ...)` which is treated as a
placeholder to replace with `shard(num_workers, worker_index)`.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
DATA<a id="DATA"></a>
</td>
<td>
`<ShardingPolicy.DATA: 3>`
</td>
</tr><tr>
<td>
DYNAMIC<a id="DYNAMIC"></a>
</td>
<td>
`<ShardingPolicy.DYNAMIC: 1>`
</td>
</tr><tr>
<td>
FILE<a id="FILE"></a>
</td>
<td>
`<ShardingPolicy.FILE: 2>`
</td>
</tr><tr>
<td>
FILE_OR_DATA<a id="FILE_OR_DATA"></a>
</td>
<td>
`<ShardingPolicy.FILE_OR_DATA: 4>`
</td>
</tr><tr>
<td>
HINT<a id="HINT"></a>
</td>
<td>
`<ShardingPolicy.HINT: 5>`
</td>
</tr><tr>
<td>
OFF<a id="OFF"></a>
</td>
<td>
`<ShardingPolicy.OFF: 0>`
</td>
</tr>
</table>

