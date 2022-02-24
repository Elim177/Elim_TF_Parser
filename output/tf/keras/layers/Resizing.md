description: A preprocessing layer which resizes images.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.Resizing" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf.keras.layers.Resizing

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/preprocessing/image_preprocessing.py#L56-L122">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A preprocessing layer which resizes images.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.experimental.preprocessing.Resizing`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.layers.Resizing`, `tf.compat.v1.keras.layers.experimental.preprocessing.Resizing`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.Resizing(
    height, width, interpolation=&#x27;bilinear&#x27;, crop_to_aspect_ratio=(False),
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer resizes an image input to a target height and width. The input
should be a 4D (batched) or 3D (unbatched) tensor in `"channels_last"` format.

For an overview and full list of preprocessing layers, see the preprocessing
[guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`height`
</td>
<td>
Integer, the height of the output shape.
</td>
</tr><tr>
<td>
`width`
</td>
<td>
Integer, the width of the output shape.
</td>
</tr><tr>
<td>
`interpolation`
</td>
<td>
String, the interpolation method. Defaults to `"bilinear"`.
Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`, `"lanczos3"`,
`"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
</td>
</tr><tr>
<td>
`crop_to_aspect_ratio`
</td>
<td>
If True, resize the images without aspect
ratio distortion. When the original aspect ratio differs from the target
aspect ratio, the output image will be cropped so as to return the largest
possible window in the image (of size `(height, width)`) that matches
the target aspect ratio. By default (`crop_to_aspect_ratio=False`),
aspect ratio may not be preserved.
</td>
</tr>
</table>



