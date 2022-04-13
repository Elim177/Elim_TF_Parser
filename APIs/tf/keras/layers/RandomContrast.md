description: A preprocessing layer which randomly adjusts contrast during training.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.RandomContrast" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf.keras.layers.RandomContrast

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/preprocessing/image_preprocessing.py#L1083-L1159">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A preprocessing layer which randomly adjusts contrast during training.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.experimental.preprocessing.RandomContrast`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.layers.RandomContrast`, `tf.compat.v1.keras.layers.experimental.preprocessing.RandomContrast`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.RandomContrast(
    factor, seed=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer will randomly adjust the contrast of an image or images by a random
factor. Contrast is adjusted independently for each channel of each image
during training.

For each channel, this layer computes the mean of the image pixels in the
channel and then adjusts each component `x` of each pixel to
`(x - mean) * contrast_factor + mean`.

For an overview and full list of preprocessing layers, see the preprocessing
[guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

#### Input shape:

3D (unbatched) or 4D (batched) tensor with shape:
`(..., height, width, channels)`, in `"channels_last"` format.



#### Output shape:

3D (unbatched) or 4D (batched) tensor with shape:
`(..., height, width, channels)`, in `"channels_last"` format.





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`factor`
</td>
<td>
a positive float represented as fraction of value, or a tuple of
size 2 representing lower and upper bound. When represented as a single
float, lower = upper. The contrast factor will be randomly picked between
`[1.0 - lower, 1.0 + upper]`.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
Integer. Used to create a random seed.
</td>
</tr>
</table>



