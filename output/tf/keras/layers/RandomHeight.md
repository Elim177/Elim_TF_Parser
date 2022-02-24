description: A preprocessing layer which randomly varies image height during training.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.RandomHeight" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf.keras.layers.RandomHeight

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/preprocessing/image_preprocessing.py#L1162-L1262">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A preprocessing layer which randomly varies image height during training.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.experimental.preprocessing.RandomHeight`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.layers.RandomHeight`, `tf.compat.v1.keras.layers.experimental.preprocessing.RandomHeight`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.RandomHeight(
    factor, interpolation=&#x27;bilinear&#x27;, seed=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer adjusts the height of a batch of images by a random factor.
The input should be a 3D (unbatched) or 4D (batched) tensor in the
`"channels_last"` image data format.

By default, this layer is inactive during inference.

For an overview and full list of preprocessing layers, see the preprocessing
[guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`factor`
</td>
<td>
A positive float (fraction of original height), or a tuple of size 2
representing lower and upper bound for resizing vertically. When
represented as a single float, this value is used for both the upper and
lower bound. For instance, `factor=(0.2, 0.3)` results in an output with
height changed by a random amount in the range `[20%, 30%]`.
`factor=(-0.2, 0.3)` results in an output with height changed by a random
amount in the range `[-20%, +30%]. `factor=0.2` results in an output with
height changed by a random amount in the range `[-20%, +20%]`.
</td>
</tr><tr>
<td>
`interpolation`
</td>
<td>
String, the interpolation method. Defaults to `"bilinear"`.
Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`,
`"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
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



#### Input shape:

3D (unbatched) or 4D (batched) tensor with shape:
`(..., height, width, channels)`, in `"channels_last"` format.



#### Output shape:

3D (unbatched) or 4D (batched) tensor with shape:
`(..., random_height, width, channels)`.


