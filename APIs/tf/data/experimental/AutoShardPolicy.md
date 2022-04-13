description: Represents the type of auto-sharding to use.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.AutoShardPolicy" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="AUTO"/>
<meta itemprop="property" content="DATA"/>
<meta itemprop="property" content="FILE"/>
<meta itemprop="property" content="HINT"/>
<meta itemprop="property" content="OFF"/>
</div>

# tf.data.experimental.AutoShardPolicy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/data/ops/options.py">View source</a>



Represents the type of auto-sharding to use.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.data.experimental.AutoShardPolicy`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

OFF: No sharding will be performed.

AUTO: Attempts FILE-based sharding, falling back to DATA-based sharding.

FILE: Shards by input files (i.e. each worker will get a set of files to
process). When this option is selected, make sure that there is at least as
many files as workers. If there are fewer input files than workers, a runtime
error will be raised.

DATA: Shards by elements produced by the dataset. Each worker will process the
whole dataset and discard the portion that is not for itself. Note that for
this mode to correctly partitions the dataset elements, the dataset needs to
produce elements in a deterministic order.

HINT: Looks for the presence of `shard(SHARD_HINT, ...)` which is treated as a
placeholder to replace with `shard(num_workers, worker_index)`.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
AUTO<a id="AUTO"></a>
</td>
<td>
`<AutoShardPolicy.AUTO: 0>`
</td>
</tr><tr>
<td>
DATA<a id="DATA"></a>
</td>
<td>
`<AutoShardPolicy.DATA: 2>`
</td>
</tr><tr>
<td>
FILE<a id="FILE"></a>
</td>
<td>
`<AutoShardPolicy.FILE: 1>`
</td>
</tr><tr>
<td>
HINT<a id="HINT"></a>
</td>
<td>
`<AutoShardPolicy.HINT: 3>`
</td>
</tr><tr>
<td>
OFF<a id="OFF"></a>
</td>
<td>
`<AutoShardPolicy.OFF: -1>`
</td>
</tr>
</table>

