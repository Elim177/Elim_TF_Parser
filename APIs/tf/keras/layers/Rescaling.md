description: A preprocessing layer which rescales input values to a new range.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.Rescaling" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf.keras.layers.Rescaling

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/preprocessing/image_preprocessing.py#L269-L322">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A preprocessing layer which rescales input values to a new range.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.experimental.preprocessing.Rescaling`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.layers.Rescaling`, `tf.compat.v1.keras.layers.experimental.preprocessing.Rescaling`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.Rescaling(
    scale, offset=0.0, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer rescales every value of an input (often an image) by multiplying by
`scale` and adding `offset`.

#### For instance:



1. To rescale an input in the `[0, 255]` range
to be in the `[0, 1]` range, you would pass `scale=1./255`.

2. To rescale an input in the `[0, 255]` range to be in the `[-1, 1]` range,
you would pass `scale=1./127.5, offset=-1`.

The rescaling is applied both during training and inference.

For an overview and full list of preprocessing layers, see the preprocessing
[guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

#### Input shape:

Arbitrary.



#### Output shape:

Same as input.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`scale`
</td>
<td>
Float, the scale to apply to the inputs.
</td>
</tr><tr>
<td>
`offset`
</td>
<td>
Float, the offset to apply to the inputs.
</td>
</tr>
</table>



