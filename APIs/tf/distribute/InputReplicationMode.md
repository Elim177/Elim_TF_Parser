description: Replication mode for input function.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.distribute.InputReplicationMode" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="PER_REPLICA"/>
<meta itemprop="property" content="PER_WORKER"/>
</div>

# tf.distribute.InputReplicationMode

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/distribute/distribute_lib.py">View source</a>



Replication mode for input function.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.distribute.InputReplicationMode`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

* `PER_WORKER`: The input function will be called on each worker
  independently, creating as many input pipelines as number of workers.
  Replicas will dequeue from the local Dataset on their worker.
  <a href="../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a> doesn't manage any state sharing between such
  separate input pipelines.
* `PER_REPLICA`: The input function will be called on each replica separately.
  <a href="../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a> doesn't manage any state sharing between such
  separate input pipelines.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
PER_REPLICA<a id="PER_REPLICA"></a>
</td>
<td>
`<InputReplicationMode.PER_REPLICA: 'PER_REPLICA'>`
</td>
</tr><tr>
<td>
PER_WORKER<a id="PER_WORKER"></a>
</td>
<td>
`<InputReplicationMode.PER_WORKER: 'PER_WORKER'>`
</td>
</tr>
</table>

