description: Returns a tensor holding Sobel edge maps.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.image.sobel_edges" />
<meta itemprop="path" content="Stable" />
</div>

# tf.image.sobel_edges

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/image_ops_impl.py">View source</a>



Returns a tensor holding Sobel edge maps.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.image.sobel_edges`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.image.sobel_edges(
    image
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Example usage:



For general usage, `image` would be loaded from a file as below:

```python
image_bytes = tf.io.read_file(path_to_image_file)
image = tf.image.decode_image(image_bytes)
image = tf.cast(image, tf.float32)
image = tf.expand_dims(image, 0)
```
But for demo purposes, we are using randomly generated values for `image`:

```
>>> image = tf.random.uniform(
...   maxval=255, shape=[1, 28, 28, 3], dtype=tf.float32)
>>> sobel = tf.image.sobel_edges(image)
>>> sobel_y = np.asarray(sobel[0, :, :, :, 0]) # sobel in y-direction
>>> sobel_x = np.asarray(sobel[0, :, :, :, 1]) # sobel in x-direction
```

For displaying the sobel results, PIL's [Image Module](
https://pillow.readthedocs.io/en/stable/reference/Image.html) can be used:

```python
# Display edge maps for the first channel (at index 0)
Image.fromarray(sobel_y[..., 0] / 4 + 0.5).show()
Image.fromarray(sobel_x[..., 0] / 4 + 0.5).show()
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
Image tensor with shape [batch_size, h, w, d] and type float32 or
float64.  The image(s) must be 2x2 or larger.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Tensor holding edge maps for each channel. Returns a tensor with shape
[batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],
[dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Sobel filter.
</td>
</tr>

</table>

