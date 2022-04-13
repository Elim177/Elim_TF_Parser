description: Cross device communication implementation.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.distribute.experimental.CommunicationImplementation" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="AUTO"/>
<meta itemprop="property" content="NCCL"/>
<meta itemprop="property" content="RING"/>
</div>

# tf.distribute.experimental.CommunicationImplementation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/distribute/collective_util.py">View source</a>



Cross device communication implementation.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.distribute.experimental.CollectiveCommunication`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.distribute.experimental.CollectiveCommunication`, `tf.compat.v1.distribute.experimental.CommunicationImplementation`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

Warning: The alias <a href="../../../tf/distribute/experimental/CommunicationImplementation.md"><code>tf.distribute.experimental.CollectiveCommunication</code></a> is
deprecated and will be removed in a future version. Use
<a href="../../../tf/distribute/experimental/CommunicationImplementation.md"><code>tf.distribute.experimental.CommunicationImplementation</code></a> instead.

* `AUTO`: Automatically chosen by Tensorflow.
* `RING`: TensorFlow's ring algorithms for all-reduce and
  all-gather.
* `NCCL`: NVIDIA®'s NCCL library. This is now only used for all-reduce on
  GPUs; all-reduce on CPU, all-gather and broadcast fallbacks to RING.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
AUTO<a id="AUTO"></a>
</td>
<td>
`<CommunicationImplementation.AUTO: 'AUTO'>`
</td>
</tr><tr>
<td>
NCCL<a id="NCCL"></a>
</td>
<td>
`<CommunicationImplementation.NCCL: 'NCCL'>`
</td>
</tr><tr>
<td>
RING<a id="RING"></a>
</td>
<td>
`<CommunicationImplementation.RING: 'RING'>`
</td>
</tr>
</table>

