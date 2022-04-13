description: Randomly crops a tensor to a given size.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.image.random_crop" />
<meta itemprop="path" content="Stable" />
</div>

# tf.image.random_crop

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/random_ops.py">View source</a>



Randomly crops a tensor to a given size.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.image.random_crop`, `tf.compat.v1.random_crop`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.image.random_crop(
    value, size, seed=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Slices a shape `size` portion out of `value` at a uniformly chosen offset.
Requires `value.shape >= size`.

If a dimension should not be cropped, pass the full size of that dimension.
For example, RGB images can be cropped with
`size = [crop_height, crop_width, 3]`.

#### Example usage:



```
>>> image = [[1, 2, 3], [4, 5, 6]]
>>> result = tf.image.random_crop(value=image, size=(1, 3))
>>> result.shape.as_list()
[1, 3]
```

For producing deterministic results given a `seed` value, use
<a href="../../tf/image/stateless_random_crop.md"><code>tf.image.stateless_random_crop</code></a>. Unlike using the `seed` param with
`tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the same
results given the same seed independent of how many times the function is
called, and independent of global seed settings (e.g. tf.random.set_seed).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`value`
</td>
<td>
Input tensor to crop.
</td>
</tr><tr>
<td>
`size`
</td>
<td>
1-D tensor with size the rank of `value`.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
Python integer. Used to create a random seed. See
<a href="../../tf/random/set_seed.md"><code>tf.random.set_seed</code></a>
for behavior.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for this operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A cropped tensor of the same rank as `value` and shape `size`.
</td>
</tr>

</table>

