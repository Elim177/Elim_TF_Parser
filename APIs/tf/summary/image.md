description: Write an image summary.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.summary.image" />
<meta itemprop="path" content="Stable" />
</div>

# tf.summary.image

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tensorboard/tree/2.7.0/tensorboard/plugins/image/summary_v2.py#L27-L142">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Write an image summary.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.summary.image(
    name, data, step=None, max_outputs=3, description=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

See also <a href="../../tf/summary/scalar.md"><code>tf.summary.scalar</code></a>, <a href="../../tf/summary/SummaryWriter.md"><code>tf.summary.SummaryWriter</code></a>.

Writes a collection of images to the current default summary writer. Data
appears in TensorBoard's 'Images' dashboard. Like <a href="../../tf/summary/scalar.md"><code>tf.summary.scalar</code></a> points,
each collection of images is associated with a `step` and a `name`.  All the
image collections with the same `name` constitute a time series of image
collections.

This example writes 2 random grayscale images:

```python
w = tf.summary.create_file_writer('test/logs')
with w.as_default():
  image1 = tf.random.uniform(shape=[8, 8, 1])
  image2 = tf.random.uniform(shape=[8, 8, 1])
  tf.summary.image("grayscale_noise", [image1, image2], step=0)
```

To avoid clipping, data should be converted to one of the following:

- floating point values in the range [0,1], or
- uint8 values in the range [0,255]

```python
# Convert the original dtype=int32 `Tensor` into `dtype=float64`.
rgb_image_float = tf.constant([
  [[1000, 0, 0], [0, 500, 1000]],
]) / 1000
tf.summary.image("picture", [rgb_image_float], step=0)

# Convert original dtype=uint8 `Tensor` into proper range.
rgb_image_uint8 = tf.constant([
  [[1, 1, 0], [0, 0, 1]],
], dtype=tf.uint8) * 255
tf.summary.image("picture", [rgb_image_uint8], step=1)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Arguments</h2></th></tr>

<tr>
<td>
`name`
</td>
<td>
A name for this summary. The summary tag used for TensorBoard will
be this name prefixed by any active name scopes.
</td>
</tr><tr>
<td>
`data`
</td>
<td>
A `Tensor` representing pixel data with shape `[k, h, w, c]`,
where `k` is the number of images, `h` and `w` are the height and
width of the images, and `c` is the number of channels, which
should be 1, 2, 3, or 4 (grayscale, grayscale with alpha, RGB, RGBA).
Any of the dimensions may be statically unknown (i.e., `None`).
Floating point data will be clipped to the range [0,1]. Other data types
will be clipped into an allowed range for safe casting to uint8, using
<a href="../../tf/image/convert_image_dtype.md"><code>tf.image.convert_image_dtype</code></a>.
</td>
</tr><tr>
<td>
`step`
</td>
<td>
Explicit `int64`-castable monotonic step value for this summary. If
omitted, this defaults to <a href="../../tf/summary/experimental/get_step.md"><code>tf.summary.experimental.get_step()</code></a>, which must
not be None.
</td>
</tr><tr>
<td>
`max_outputs`
</td>
<td>
Optional `int` or rank-0 integer `Tensor`. At most this
many images will be emitted at each step. When more than
`max_outputs` many images are provided, the first `max_outputs` many
images will be used and the rest silently discarded.
</td>
</tr><tr>
<td>
`description`
</td>
<td>
Optional long-form description for this summary, as a
constant `str`. Markdown is supported. Defaults to empty.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
True on success, or false if no summary was emitted because no default
summary writer was available.
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
if a default writer exists, but no step was provided and
<a href="../../tf/summary/experimental/get_step.md"><code>tf.summary.experimental.get_step()</code></a> is None.
</td>
</tr>
</table>

