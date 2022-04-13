description: Crops an image to a specified bounding box.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.image.crop_to_bounding_box" />
<meta itemprop="path" content="Stable" />
</div>

# tf.image.crop_to_bounding_box

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/image_ops_impl.py">View source</a>



Crops an `image` to a specified bounding box.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.image.crop_to_bounding_box`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.image.crop_to_bounding_box(
    image, offset_height, offset_width, target_height, target_width
)
</code></pre>



<!-- Placeholder for "Used in" -->

This op cuts a rectangular bounding box out of `image`. The top-left corner
of the bounding box is at `offset_height, offset_width` in `image`, and the
lower-right corner is at
`offset_height + target_height, offset_width + target_width`.

#### Example Usage:



```
>>> image = tf.constant(np.arange(1, 28, dtype=np.float32), shape=[3, 3, 3])
>>> image[:,:,0] # print the first channel of the 3-D tensor
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[ 1.,  4.,  7.],
       [10., 13., 16.],
       [19., 22., 25.]], dtype=float32)>
>>> cropped_image = tf.image.crop_to_bounding_box(image, 0, 0, 2, 2)
>>> cropped_image[:,:,0] # print the first channel of the cropped 3-D tensor
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[ 1.,  4.],
       [10., 13.]], dtype=float32)>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`image`
</td>
<td>
4-D `Tensor` of shape `[batch, height, width, channels]` or 3-D
`Tensor` of shape `[height, width, channels]`.
</td>
</tr><tr>
<td>
`offset_height`
</td>
<td>
Vertical coordinate of the top-left corner of the bounding
box in `image`.
</td>
</tr><tr>
<td>
`offset_width`
</td>
<td>
Horizontal coordinate of the top-left corner of the bounding
box in `image`.
</td>
</tr><tr>
<td>
`target_height`
</td>
<td>
Height of the bounding box.
</td>
</tr><tr>
<td>
`target_width`
</td>
<td>
Width of the bounding box.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
If `image` was 4-D, a 4-D `Tensor` of shape
`[batch, target_height, target_width, channels]`.
If `image` was 3-D, a 3-D `Tensor` of shape
`[target_height, target_width, channels]`.
It has the same dtype with `image`.
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
`image` is not a 3-D or 4-D `Tensor`.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
`offset_width < 0` or `offset_height < 0`.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
`target_width <= 0` or `target_width <= 0`.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
`width < offset_width + target_width` or
`height < offset_height + target_height`.
</td>
</tr>
</table>

