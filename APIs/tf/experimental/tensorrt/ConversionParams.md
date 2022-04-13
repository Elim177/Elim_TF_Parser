description: Parameters that are used for TF-TRT conversion.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.tensorrt.ConversionParams" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="allow_build_at_runtime"/>
<meta itemprop="property" content="max_workspace_size_bytes"/>
<meta itemprop="property" content="maximum_cached_engines"/>
<meta itemprop="property" content="minimum_segment_size"/>
<meta itemprop="property" content="precision_mode"/>
<meta itemprop="property" content="use_calibration"/>
</div>

# tf.experimental.tensorrt.ConversionParams

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/compiler/tensorrt/trt_convert.py">View source</a>



Parameters that are used for TF-TRT conversion.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.tensorrt.ConversionParams(
    max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
    precision_mode=TrtPrecisionMode.FP32, minimum_segment_size=3,
    maximum_cached_engines=1, use_calibration=(True), allow_build_at_runtime=(True)
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Fields:


* <b>`max_workspace_size_bytes`</b>: the maximum GPU temporary memory which the TRT
  engine can use at execution time. This corresponds to the
  'workspaceSize' parameter of nvinfer1::IBuilder::setMaxWorkspaceSize().
* <b>`precision_mode`</b>: one the strings in
  TrtPrecisionMode.supported_precision_modes().
* <b>`minimum_segment_size`</b>: the minimum number of nodes required for a subgraph
  to be replaced by TRTEngineOp.
* <b>`maximum_cached_engines`</b>: max number of cached TRT engines for dynamic TRT
  ops. Created TRT engines for a dynamic dimension are cached. This is the
  maximum number of engines that can be cached. If the number of cached
  engines is already at max but none of them supports the input shapes,
  the TRTEngineOp will fall back to run the original TF subgraph that
  corresponds to the TRTEngineOp.
* <b>`use_calibration`</b>: this argument is ignored if precision_mode is not INT8.
  If set to True, a calibration graph will be created to calibrate the
  missing ranges. The calibration graph must be converted to an inference
  graph by running calibration with calibrate(). If set to False,
  quantization nodes will be expected for every tensor in the graph
  (excluding those which will be fused). If a range is missing, an error
  will occur. Please note that accuracy may be negatively affected if
  there is a mismatch between which tensors TRT quantizes and which
  tensors were trained with fake quantization.
* <b>`allow_build_at_runtime`</b>: whether to build TensorRT engines during runtime.
  If no TensorRT engine can be found in cache that can handle the given
  inputs during runtime, then a new TensorRT engine is built at runtime if
  allow_build_at_runtime=True, and otherwise native TF is used.




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
allow_build_at_runtime<a id="allow_build_at_runtime"></a>
</td>
<td>
Instance of `_collections._tuplegetter`

Alias for field number 5
</td>
</tr><tr>
<td>
max_workspace_size_bytes<a id="max_workspace_size_bytes"></a>
</td>
<td>
Instance of `_collections._tuplegetter`

Alias for field number 0
</td>
</tr><tr>
<td>
maximum_cached_engines<a id="maximum_cached_engines"></a>
</td>
<td>
Instance of `_collections._tuplegetter`

Alias for field number 3
</td>
</tr><tr>
<td>
minimum_segment_size<a id="minimum_segment_size"></a>
</td>
<td>
Instance of `_collections._tuplegetter`

Alias for field number 2
</td>
</tr><tr>
<td>
precision_mode<a id="precision_mode"></a>
</td>
<td>
Instance of `_collections._tuplegetter`

Alias for field number 1
</td>
</tr><tr>
<td>
use_calibration<a id="use_calibration"></a>
</td>
<td>
Instance of `_collections._tuplegetter`

Alias for field number 4
</td>
</tr>
</table>
