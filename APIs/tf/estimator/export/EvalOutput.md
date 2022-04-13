description: Represents the output of a supervised eval process.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.estimator.export.EvalOutput" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="as_signature_def"/>
<meta itemprop="property" content="LOSS_NAME"/>
<meta itemprop="property" content="METRICS_NAME"/>
<meta itemprop="property" content="METRIC_UPDATE_SUFFIX"/>
<meta itemprop="property" content="METRIC_VALUE_SUFFIX"/>
<meta itemprop="property" content="PREDICTIONS_NAME"/>
</div>

# tf.estimator.export.EvalOutput

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/saved_model/model_utils/export_output.py">View source</a>



Represents the output of a supervised eval process.

Inherits From: [`ExportOutput`](../../../tf/estimator/export/ExportOutput.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.estimator.export.EvalOutput`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.estimator.export.EvalOutput(
    loss=None, predictions=None, metrics=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This class generates the appropriate signature def for exporting
eval output by type-checking and wrapping loss, predictions, and metrics
values.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`loss`
</td>
<td>
dict of Tensors or single Tensor representing calculated loss.
</td>
</tr><tr>
<td>
`predictions`
</td>
<td>
dict of Tensors or single Tensor representing model
predictions.
</td>
</tr><tr>
<td>
`metrics`
</td>
<td>
Dict of metric results keyed by name.
The values of the dict can be one of the following:
(1) instance of `Metric` class.
(2) (metric_value, update_op) tuples, or a single tuple.
metric_value must be a Tensor, and update_op must be a Tensor or Op.
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
if any of the outputs' dict keys are not strings or tuples of
strings or the values are not Tensors (or Operations in the case of
update_op).
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`loss`
</td>
<td>

</td>
</tr><tr>
<td>
`metrics`
</td>
<td>

</td>
</tr><tr>
<td>
`predictions`
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="as_signature_def"><code>as_signature_def</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/saved_model/model_utils/export_output.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_signature_def(
    receiver_tensors
)
</code></pre>

Generate a SignatureDef proto for inclusion in a MetaGraphDef.

The SignatureDef will specify outputs as described in this ExportOutput,
and will use the provided receiver_tensors as inputs.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`receiver_tensors`
</td>
<td>
a `Tensor`, or a dict of string to `Tensor`, specifying
input nodes that will be fed.
</td>
</tr>
</table>







<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
LOSS_NAME<a id="LOSS_NAME"></a>
</td>
<td>
`'loss'`
</td>
</tr><tr>
<td>
METRICS_NAME<a id="METRICS_NAME"></a>
</td>
<td>
`'metrics'`
</td>
</tr><tr>
<td>
METRIC_UPDATE_SUFFIX<a id="METRIC_UPDATE_SUFFIX"></a>
</td>
<td>
`'update_op'`
</td>
</tr><tr>
<td>
METRIC_VALUE_SUFFIX<a id="METRIC_VALUE_SUFFIX"></a>
</td>
<td>
`'value'`
</td>
</tr><tr>
<td>
PREDICTIONS_NAME<a id="PREDICTIONS_NAME"></a>
</td>
<td>
`'predictions'`
</td>
</tr>
</table>

