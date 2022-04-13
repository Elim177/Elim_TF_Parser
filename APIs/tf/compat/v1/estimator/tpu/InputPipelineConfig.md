description: Please see the definition of these values in TPUConfig.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.estimator.tpu.InputPipelineConfig" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="BROADCAST"/>
<meta itemprop="property" content="PER_HOST_V1"/>
<meta itemprop="property" content="PER_HOST_V2"/>
<meta itemprop="property" content="PER_SHARD_V1"/>
<meta itemprop="property" content="SLICED"/>
</div>

# tf.compat.v1.estimator.tpu.InputPipelineConfig

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator/python/estimator/tpu/tpu_config.py#L36-L51">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Please see the definition of these values in TPUConfig.



 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

TPU Estimator manages its own TensorFlow graph and session, so it is not
compatible with TF2 behaviors. We recommend that you migrate to the newer
<a href="../../../../../tf/distribute/TPUStrategy.md"><code>tf.distribute.TPUStrategy</code></a>. See the
[TPU guide](https://www.tensorflow.org/guide/tpu) for details.


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
BROADCAST<a id="BROADCAST"></a>
</td>
<td>
`4`
</td>
</tr><tr>
<td>
PER_HOST_V1<a id="PER_HOST_V1"></a>
</td>
<td>
`2`
</td>
</tr><tr>
<td>
PER_HOST_V2<a id="PER_HOST_V2"></a>
</td>
<td>
`3`
</td>
</tr><tr>
<td>
PER_SHARD_V1<a id="PER_SHARD_V1"></a>
</td>
<td>
`1`
</td>
</tr><tr>
<td>
SLICED<a id="SLICED"></a>
</td>
<td>
`5`
</td>
</tr>
</table>

