description: Mapping from logical cores in a computation to the physical TPU topology.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.tpu.experimental.DeviceAssignment" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="coordinates"/>
<meta itemprop="property" content="host_device"/>
<meta itemprop="property" content="lookup_replicas"/>
<meta itemprop="property" content="tpu_device"/>
<meta itemprop="property" content="tpu_ordinal"/>
</div>

# tf.tpu.experimental.DeviceAssignment

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/tpu/device_assignment.py">View source</a>



Mapping from logical cores in a computation to the physical TPU topology.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.tpu.experimental.DeviceAssignment`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.tpu.experimental.DeviceAssignment(
    topology: <a href="../../../tf/tpu/experimental/Topology.md"><code>tf.tpu.experimental.Topology</code></a>,
    core_assignment: np.ndarray
)
</code></pre>



<!-- Placeholder for "Used in" -->

Prefer to use the <a href="../../../tf/tpu/experimental/DeviceAssignment.md#build"><code>DeviceAssignment.build()</code></a> helper to construct a
`DeviceAssignment`; it is easier if less flexible than constructing a
`DeviceAssignment` directly.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`topology`
</td>
<td>
A `Topology` object that describes the physical TPU topology.
</td>
</tr><tr>
<td>
`core_assignment`
</td>
<td>
A logical to physical core mapping, represented as a
rank 3 numpy array. See the description of the `core_assignment`
property for more details.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If `topology` is not `Topology` object.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If `core_assignment` is not a rank 3 numpy array.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`core_assignment`
</td>
<td>
The logical to physical core mapping.
</td>
</tr><tr>
<td>
`num_cores_per_replica`
</td>
<td>
The number of cores per replica.
</td>
</tr><tr>
<td>
`num_replicas`
</td>
<td>
The number of replicas of the computation.
</td>
</tr><tr>
<td>
`topology`
</td>
<td>
A `Topology` that describes the TPU topology.
</td>
</tr>
</table>



## Methods

<h3 id="build"><code>build</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/tpu/device_assignment.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>build(
    topology: <a href="../../../tf/tpu/experimental/Topology.md"><code>tf.tpu.experimental.Topology</code></a>,
    computation_shape: Optional[np.ndarray] = None,
    computation_stride: Optional[np.ndarray] = None,
    num_replicas: int = 1
) -> "DeviceAssignment"
</code></pre>




<h3 id="coordinates"><code>coordinates</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/tpu/device_assignment.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>coordinates(
    replica: int,
    logical_core: int
) -> Tuple
</code></pre>

Returns the physical topology coordinates of a logical core.


<h3 id="host_device"><code>host_device</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/tpu/device_assignment.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>host_device(
    replica: int = 0,
    logical_core: int = 0,
    job: Optional[Text] = None
) -> Text
</code></pre>

Returns the CPU device attached to a logical core.


<h3 id="lookup_replicas"><code>lookup_replicas</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/tpu/device_assignment.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>lookup_replicas(
    task_id: int,
    logical_core: int
) -> List[int]
</code></pre>

Lookup replica ids by task number and logical core.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`task_id`
</td>
<td>
TensorFlow task number.
</td>
</tr><tr>
<td>
`logical_core`
</td>
<td>
An integer, identifying a logical core.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A sorted list of the replicas that are attached to that task and
logical_core.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If no replica exists in the task which contains the logical
core.
</td>
</tr>
</table>



<h3 id="tpu_device"><code>tpu_device</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/tpu/device_assignment.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tpu_device(
    replica: int = 0,
    logical_core: int = 0,
    job: Optional[Text] = None
) -> Text
</code></pre>

Returns the name of the TPU device assigned to a logical core.


<h3 id="tpu_ordinal"><code>tpu_ordinal</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/tpu/device_assignment.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tpu_ordinal(
    replica: int = 0,
    logical_core: int = 0
) -> int
</code></pre>

Returns the ordinal of the TPU device assigned to a logical core.




