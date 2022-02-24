description: Representative dataset used to optimize the model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.lite.RepresentativeDataset" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.lite.RepresentativeDataset

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/lite/python/lite.py">View source</a>



Representative dataset used to optimize the model.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.lite.RepresentativeDataset`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.lite.RepresentativeDataset(
    input_gen
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is a generator function that provides a small dataset to calibrate or
estimate the range, i.e, (min, max) of all floating-point arrays in the model
(such as model input, activation outputs of intermediate layers, and model
output) for quantization. Usually, this is a small subset of a few hundred
samples randomly chosen, in no particular order, from the training or
evaluation dataset.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_gen`
</td>
<td>
A generator function that generates input samples for the
model and has the same order, type and shape as the inputs to the model.
Usually, this is a small subset of a few hundred samples randomly
chosen, in no particular order, from the training or evaluation dataset.
</td>
</tr>
</table>



