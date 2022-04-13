description: Provides a collection of TFLite model analyzer tools.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.lite.experimental.Analyzer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="analyze"/>
</div>

# tf.lite.experimental.Analyzer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/lite/python/analyzer.py">View source</a>



Provides a collection of TFLite model analyzer tools.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.lite.experimental.Analyzer`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->


#### Example:



```python
model = tf.keras.applications.MobileNetV3Large()
fb_model = tf.lite.TFLiteConverterV2.from_keras_model(model).convert()
tf.lite.experimental.Analyzer.analyze(model_content=fb_model)
# === TFLite ModelAnalyzer ===
#
# Your TFLite model has ‘1’ subgraph(s). In the subgraph description below,
# T# represents the Tensor numbers. For example, in Subgraph#0, the MUL op
# takes tensor #0 and tensor #19 as input and produces tensor #136 as output.
#
# Subgraph#0 main(T#0) -> [T#263]
#   Op#0 MUL(T#0, T#19) -> [T#136]
#   Op#1 ADD(T#136, T#18) -> [T#137]
#   Op#2 CONV_2D(T#137, T#44, T#93) -> [T#138]
#   Op#3 HARD_SWISH(T#138) -> [T#139]
#   Op#4 DEPTHWISE_CONV_2D(T#139, T#94, T#24) -> [T#140]
#   ...
```

WARNING: Experimental interface, subject to change.

## Methods

<h3 id="analyze"><code>analyze</code></h3>

<a target="_blank" href="/code/stable/tensorflow/lite/python/analyzer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>analyze(
    model_path=None, model_content=None, gpu_compatibility=(False), **kwargs
)
</code></pre>

Analyzes the given tflite_model with dumping model structure.

This tool provides a way to understand users' TFLite flatbuffer model by
dumping internal graph structure. It also provides additional features
like checking GPU delegate compatibility.

WARNING: Experimental interface, subject to change.
WARNING: The output format is not guaranteed to stay stable, so don't
         write scripts to this.
Args:
  model_path: TFLite flatbuffer model path.
  model_content: TFLite flatbuffer model object.
  gpu_compatibility: Whether to check GPU delegate compatibility.
  **kwargs: Experimental keyword arguments to analyze API.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Print analyzed report via console output.
</td>
</tr>

</table>





