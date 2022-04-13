description: Partitioner to specify a fixed number of shards along given axis.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.fixed_size_partitioner" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.fixed_size_partitioner

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/partitioned_variables.py">View source</a>



Partitioner to specify a fixed number of shards along given axis.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.fixed_size_partitioner(
    num_shards, axis=0
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

This API is deprecated in TF2. In TF2, partitioner is no longer part of
the variable declaration via <a href="../../../tf/Variable.md"><code>tf.Variable</code></a>.
[ParameterServer Training]
(https://www.tensorflow.org/tutorials/distribute/parameter_server_training)
handles partitioning of variables. The corresponding TF2 partitioner class of
`fixed_size_partitioner` is
<a href="../../../tf/distribute/experimental/partitioners/FixedShardsPartitioner.md"><code>tf.distribute.experimental.partitioners.FixedShardsPartitioner</code></a>.

Check the [migration guide]
(https://www.tensorflow.org/guide/migrate#2_use_python_objects_to_track_variables_and_losses)
on the differences in treatment of variables and losses between TF1 and TF2.

Before:

  ```
  x = tf.compat.v1.get_variable(
    "x", shape=(2,), partitioner=tf.compat.v1.fixed_size_partitioner(2)
  )
  ```
After:

  ```
  partitioner = (
      tf.distribute.experimental.partitioners.FixedShardsPartitioner(
          num_shards=2)
  )
  strategy = tf.distribute.experimental.ParameterServerStrategy(
                 cluster_resolver=cluster_resolver,
                 variable_partitioner=partitioner)

  with strategy.scope():
    x = tf.Variable([1.0, 2.0])
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
`num_shards`
</td>
<td>
`int`, number of shards to partition variable.
</td>
</tr><tr>
<td>
`axis`
</td>
<td>
`int`, axis to partition on.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A partition function usable as the `partitioner` argument to
`variable_scope` and `get_variable`.
</td>
</tr>

</table>

