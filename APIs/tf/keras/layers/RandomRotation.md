description: A preprocessing layer which randomly rotates images during training.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.RandomRotation" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf.keras.layers.RandomRotation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/preprocessing/image_preprocessing.py#L749-L873">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A preprocessing layer which randomly rotates images during training.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.experimental.preprocessing.RandomRotation`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.layers.RandomRotation`, `tf.compat.v1.keras.layers.experimental.preprocessing.RandomRotation`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.RandomRotation(
    factor, fill_mode=&#x27;reflect&#x27;, interpolation=&#x27;bilinear&#x27;,
    seed=None, fill_value=0.0, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer will apply random rotations to each image, filling empty space
according to `fill_mode`.

By default, random rotations are only applied during training.
At inference time, the layer does nothing. If you need to apply random
rotations at inference time, set `training` to True when calling the layer.

For an overview and full list of preprocessing layers, see the preprocessing
[guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

#### Input shape:

3D (unbatched) or 4D (batched) tensor with shape:
`(..., height, width, channels)`, in `"channels_last"` format



#### Output shape:

3D (unbatched) or 4D (batched) tensor with shape:
`(..., height, width, channels)`, in `"channels_last"` format





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`factor`
</td>
<td>
a float represented as fraction of 2 Pi, or a tuple of size 2
representing lower and upper bound for rotating clockwise and
counter-clockwise. A positive values means rotating counter clock-wise,
while a negative value means clock-wise. When represented as a single
float, this value is used for both the upper and lower bound. For
instance, `factor=(-0.2, 0.3)` results in an output rotation by a random
amount in the range `[-20% * 2pi, 30% * 2pi]`. `factor=0.2` results in an
output rotating by a random amount in the range `[-20% * 2pi, 20% * 2pi]`.
</td>
</tr><tr>
<td>
`fill_mode`
</td>
<td>
Points outside the boundaries of the input are filled according
to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
- *reflect*: `(d c b a | a b c d | d c b a)` The input is extended by
  reflecting about the edge of the last pixel.
- *constant*: `(k k k k | a b c d | k k k k)` The input is extended by
  filling all values beyond the edge with the same constant value k = 0.
- *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
  wrapping around to the opposite edge.
- *nearest*: `(a a a a | a b c d | d d d d)` The input is extended by the
  nearest pixel.
</td>
</tr><tr>
<td>
`interpolation`
</td>
<td>
Interpolation mode. Supported values: `"nearest"`,
`"bilinear"`.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
Integer. Used to create a random seed.
</td>
</tr><tr>
<td>
`fill_value`
</td>
<td>
a float represents the value to be filled outside the boundaries
when `fill_mode="constant"`.
</td>
</tr>
</table>



