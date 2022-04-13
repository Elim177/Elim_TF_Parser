description: Linearly scales each image in image to have mean 0 and variance 1.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.image.per_image_standardization" />
<meta itemprop="path" content="Stable" />
</div>

# tf.image.per_image_standardization

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/image_ops_impl.py">View source</a>



Linearly scales each image in `image` to have mean 0 and variance 1.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.image.per_image_standardization`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.image.per_image_standardization(
    image
)
</code></pre>



<!-- Placeholder for "Used in" -->

For each 3-D image `x` in `image`, computes `(x - mean) / adjusted_stddev`,
where

- `mean` is the average of all values in `x`
- `adjusted_stddev = max(stddev, 1.0/sqrt(N))` is capped away from 0 to
  protect against division by 0 when handling uniform images
  - `N` is the number of elements in `x`
  - `stddev` is the standard deviation of all values in `x`

#### Example Usage:



```
>>> image = tf.constant(np.arange(1, 13, dtype=np.int32), shape=[2, 2, 3])
>>> image # 3-D tensor
<tf.Tensor: shape=(2, 2, 3), dtype=int32, numpy=
array([[[ 1,  2,  3],
        [ 4,  5,  6]],
       [[ 7,  8,  9],
        [10, 11, 12]]], dtype=int32)>
>>> new_image = tf.image.per_image_standardization(image)
>>> new_image # 3-D tensor with mean ~= 0 and variance ~= 1
<tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
array([[[-1.593255  , -1.3035723 , -1.0138896 ],
        [-0.7242068 , -0.4345241 , -0.14484136]],
       [[ 0.14484136,  0.4345241 ,  0.7242068 ],
        [ 1.0138896 ,  1.3035723 ,  1.593255  ]]], dtype=float32)>
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
An n-D `Tensor` with at least 3 dimensions, the last 3 of which are
the dimensions of each image.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` with the same shape as `image` and its dtype is `float32`.
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
The shape of `image` has fewer than 3 dimensions.
</td>
</tr>
</table>

