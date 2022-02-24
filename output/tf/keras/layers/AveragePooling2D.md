description: Average pooling operation for spatial data.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.AveragePooling2D" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf.keras.layers.AveragePooling2D

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/pooling.py#L532-L643">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Average pooling operation for spatial data.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.AvgPool2D`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.layers.AveragePooling2D`, `tf.compat.v1.keras.layers.AvgPool2D`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.AveragePooling2D(
    pool_size=(2, 2), strides=None, padding=&#x27;valid&#x27;, data_format=None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Downsamples the input along its spatial dimensions (height and width)
by taking the average value over an input window
(of size defined by `pool_size`) for each channel of the input.
The window is shifted by `strides` along each dimension.

The resulting output when using `"valid"` padding option has a shape
(number of rows or columns) of:
`output_shape = math.floor((input_shape - pool_size) / strides) + 1`
(when `input_shape >= pool_size`)

The resulting output shape when using the `"same"` padding option is:
`output_shape = math.floor((input_shape - 1) / strides) + 1`

For example, for `strides=(1, 1)` and `padding="valid"`:

```
>>> x = tf.constant([[1., 2., 3.],
...                  [4., 5., 6.],
...                  [7., 8., 9.]])
>>> x = tf.reshape(x, [1, 3, 3, 1])
>>> avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
...    strides=(1, 1), padding='valid')
>>> avg_pool_2d(x)
<tf.Tensor: shape=(1, 2, 2, 1), dtype=float32, numpy=
  array([[[[3.],
           [4.]],
          [[6.],
           [7.]]]], dtype=float32)>
```

For example, for `stride=(2, 2)` and `padding="valid"`:

```
>>> x = tf.constant([[1., 2., 3., 4.],
...                  [5., 6., 7., 8.],
...                  [9., 10., 11., 12.]])
>>> x = tf.reshape(x, [1, 3, 4, 1])
>>> avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
...    strides=(2, 2), padding='valid')
>>> avg_pool_2d(x)
<tf.Tensor: shape=(1, 1, 2, 1), dtype=float32, numpy=
  array([[[[3.5],
           [5.5]]]], dtype=float32)>
```

For example, for `strides=(1, 1)` and `padding="same"`:

```
>>> x = tf.constant([[1., 2., 3.],
...                  [4., 5., 6.],
...                  [7., 8., 9.]])
>>> x = tf.reshape(x, [1, 3, 3, 1])
>>> avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
...    strides=(1, 1), padding='same')
>>> avg_pool_2d(x)
<tf.Tensor: shape=(1, 3, 3, 1), dtype=float32, numpy=
  array([[[[3.],
           [4.],
           [4.5]],
          [[6.],
           [7.],
           [7.5]],
          [[7.5],
           [8.5],
           [9.]]]], dtype=float32)>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`pool_size`
</td>
<td>
integer or tuple of 2 integers,
factors by which to downscale (vertical, horizontal).
`(2, 2)` will halve the input in both spatial dimension.
If only one integer is specified, the same window length
will be used for both dimensions.
</td>
</tr><tr>
<td>
`strides`
</td>
<td>
Integer, tuple of 2 integers, or None.
Strides values.
If None, it will default to `pool_size`.
</td>
</tr><tr>
<td>
`padding`
</td>
<td>
One of `"valid"` or `"same"` (case-insensitive).
`"valid"` means no padding. `"same"` results in padding evenly to
the left/right or up/down of the input such that output has the same
height/width dimension as the input.
</td>
</tr><tr>
<td>
`data_format`
</td>
<td>
A string,
one of `channels_last` (default) or `channels_first`.
The ordering of the dimensions in the inputs.
`channels_last` corresponds to inputs with shape
`(batch, height, width, channels)` while `channels_first`
corresponds to inputs with shape
`(batch, channels, height, width)`.
It defaults to the `image_data_format` value found in your
Keras config file at `~/.keras/keras.json`.
If you never set it, then it will be "channels_last".
</td>
</tr>
</table>



#### Input shape:

- If `data_format='channels_last'`:
  4D tensor with shape `(batch_size, rows, cols, channels)`.
- If `data_format='channels_first'`:
  4D tensor with shape `(batch_size, channels, rows, cols)`.



#### Output shape:

- If `data_format='channels_last'`:
  4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
- If `data_format='channels_first'`:
  4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.


