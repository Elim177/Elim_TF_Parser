description: A ProtocolMessage

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.ConfigProto.Experimental" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="MLIR_BRIDGE_ROLLOUT_DISABLED"/>
<meta itemprop="property" content="MLIR_BRIDGE_ROLLOUT_ENABLED"/>
<meta itemprop="property" content="MLIR_BRIDGE_ROLLOUT_SAFE_MODE_ENABLED"/>
<meta itemprop="property" content="MLIR_BRIDGE_ROLLOUT_SAFE_MODE_FALLBACK_ENABLED"/>
<meta itemprop="property" content="MLIR_BRIDGE_ROLLOUT_UNSPECIFIED"/>
<meta itemprop="property" content="MlirBridgeRollout"/>
</div>

# tf.compat.v1.ConfigProto.Experimental

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/core/protobuf/config.proto">View source</a>



A ProtocolMessage

<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`collective_deterministic_sequential_execution`
</td>
<td>
`bool collective_deterministic_sequential_execution`
</td>
</tr><tr>
<td>
`collective_group_leader`
</td>
<td>
`string collective_group_leader`
</td>
</tr><tr>
<td>
`collective_nccl`
</td>
<td>
`bool collective_nccl`
</td>
</tr><tr>
<td>
`coordination_service`
</td>
<td>
`string coordination_service`
</td>
</tr><tr>
<td>
`disable_functional_ops_lowering`
</td>
<td>
`bool disable_functional_ops_lowering`
</td>
</tr><tr>
<td>
`disable_output_partition_graphs`
</td>
<td>
`bool disable_output_partition_graphs`
</td>
</tr><tr>
<td>
`disable_thread_spinning`
</td>
<td>
`bool disable_thread_spinning`
</td>
</tr><tr>
<td>
`enable_mlir_bridge`
</td>
<td>
`bool enable_mlir_bridge`
</td>
</tr><tr>
<td>
`enable_mlir_graph_optimization`
</td>
<td>
`bool enable_mlir_graph_optimization`
</td>
</tr><tr>
<td>
`executor_type`
</td>
<td>
`string executor_type`
</td>
</tr><tr>
<td>
`mlir_bridge_rollout`
</td>
<td>
`MlirBridgeRollout mlir_bridge_rollout`
</td>
</tr><tr>
<td>
`optimize_for_static_graph`
</td>
<td>
`bool optimize_for_static_graph`
</td>
</tr><tr>
<td>
`recv_buf_max_chunk`
</td>
<td>
`int32 recv_buf_max_chunk`
</td>
</tr><tr>
<td>
`session_metadata`
</td>
<td>
`SessionMetadata session_metadata`
</td>
</tr><tr>
<td>
`share_cluster_devices_in_session`
</td>
<td>
`bool share_cluster_devices_in_session`
</td>
</tr><tr>
<td>
`share_session_state_in_clusterspec_propagation`
</td>
<td>
`bool share_session_state_in_clusterspec_propagation`
</td>
</tr><tr>
<td>
`use_numa_affinity`
</td>
<td>
`bool use_numa_affinity`
</td>
</tr><tr>
<td>
`use_tfrt`
</td>
<td>
`bool use_tfrt`
</td>
</tr><tr>
<td>
`xla_fusion_autotuner_thresh`
</td>
<td>
`int64 xla_fusion_autotuner_thresh`
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
MLIR_BRIDGE_ROLLOUT_DISABLED<a id="MLIR_BRIDGE_ROLLOUT_DISABLED"></a>
</td>
<td>
`2`
</td>
</tr><tr>
<td>
MLIR_BRIDGE_ROLLOUT_ENABLED<a id="MLIR_BRIDGE_ROLLOUT_ENABLED"></a>
</td>
<td>
`1`
</td>
</tr><tr>
<td>
MLIR_BRIDGE_ROLLOUT_SAFE_MODE_ENABLED<a id="MLIR_BRIDGE_ROLLOUT_SAFE_MODE_ENABLED"></a>
</td>
<td>
`3`
</td>
</tr><tr>
<td>
MLIR_BRIDGE_ROLLOUT_SAFE_MODE_FALLBACK_ENABLED<a id="MLIR_BRIDGE_ROLLOUT_SAFE_MODE_FALLBACK_ENABLED"></a>
</td>
<td>
`4`
</td>
</tr><tr>
<td>
MLIR_BRIDGE_ROLLOUT_UNSPECIFIED<a id="MLIR_BRIDGE_ROLLOUT_UNSPECIFIED"></a>
</td>
<td>
`0`
</td>
</tr><tr>
<td>
MlirBridgeRollout<a id="MlirBridgeRollout"></a>
</td>
<td>
Instance of `google.protobuf.internal.enum_type_wrapper.EnumTypeWrapper`
</td>
</tr>
</table>

