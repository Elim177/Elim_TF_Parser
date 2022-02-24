description: Debug options to set up a given QuantizationDebugger.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.lite.experimental.QuantizationDebugOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.lite.experimental.QuantizationDebugOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/lite/tools/optimize/debugging/python/debugger.py">View source</a>



Debug options to set up a given QuantizationDebugger.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.lite.experimental.QuantizationDebugOptions`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.lite.experimental.QuantizationDebugOptions(
    layer_debug_metrics: Optional[Mapping[str, Callable[[np.ndarray], float]]] = None,
    model_debug_metrics: Optional[Mapping[str, Callable[[Sequence[np.ndarray], Sequence[np.ndarray]],
        float]]] = None,
    layer_direct_compare_metrics: Optional[Mapping[str, Callable[[Sequence[np.ndarray], Sequence[np.ndarray],
        float, int], float]]] = None,
    denylisted_ops: Optional[List[str]] = None,
    denylisted_nodes: Optional[List[str]] = None,
    fully_quantize: bool = (False)
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`layer_debug_metrics`
</td>
<td>
a dict to specify layer debug functions
{function_name_str: function} where the function accepts result of
  NumericVerify Op, which is value difference between float and
  dequantized op results. The function returns single scalar value.
</td>
</tr><tr>
<td>
`model_debug_metrics`
</td>
<td>
a dict to specify model debug functions
{function_name_str: function} where the function accepts outputs from
  two models, and returns single scalar value for a metric. (e.g.
  accuracy, IoU)
</td>
</tr><tr>
<td>
`layer_direct_compare_metrics`
</td>
<td>
a dict to specify layer debug functions
{function_name_str: function}. The signature is different from that of
  `layer_debug_metrics`, and this one gets passed (original float value,
  original quantized value, scale, zero point). The function's
  implementation is responsible for correctly dequantize the quantized
  value to compare. Use this one when comparing diff is not enough.
  (Note) quantized value is passed as int8, so cast to int32 is needed.
</td>
</tr><tr>
<td>
`denylisted_ops`
</td>
<td>
a list of op names which is expected to be removed from
quantization.
</td>
</tr><tr>
<td>
`denylisted_nodes`
</td>
<td>
a list of op's output tensor names to be removed from
quantization.
</td>
</tr><tr>
<td>
`fully_quantize`
</td>
<td>
Bool indicating whether to fully quantize the model.
Besides model body, the input/output will be quantized as well.
Corresponding to mlir_quantize's fully_quantize parameter.
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
when there are duplicate keys
</td>
</tr>
</table>



