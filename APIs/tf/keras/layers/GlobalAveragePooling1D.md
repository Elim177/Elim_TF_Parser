description: Global average pooling operation for temporal data.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.GlobalAveragePooling1D" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf.keras.layers.GlobalAveragePooling1D

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/pooling.py#L916-L987">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Global average pooling operation for temporal data.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.GlobalAvgPool1D`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.layers.GlobalAveragePooling1D`, `tf.compat.v1.keras.layers.GlobalAvgPool1D`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.GlobalAveragePooling1D(
    data_format=&#x27;channels_last&#x27;, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Examples:



```
>>> input_shape = (2, 3, 4)
>>> x = tf.random.normal(input_shape)
>>> y = tf.keras.layers.GlobalAveragePooling1D()(x)
>>> print(y.shape)
(2, 4)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`data_format`
</td>
<td>
A string,
one of `channels_last` (default) or `channels_first`.
The ordering of the dimensions in the inputs.
`channels_last` corresponds to inputs with shape
`(batch, steps, features)` while `channels_first`
corresponds to inputs with shape
`(batch, features, steps)`.
</td>
</tr><tr>
<td>
`keepdims`
</td>
<td>
A boolean, whether to keep the temporal dimension or not.
If `keepdims` is `False` (default), the rank of the tensor is reduced
for spatial dimensions.
If `keepdims` is `True`, the temporal dimension are retained with
length 1.
The behavior is the same as for <a href="../../../tf/math/reduce_mean.md"><code>tf.reduce_mean</code></a> or `np.mean`.
</td>
</tr>
</table>



#### Call arguments:


* <b>`inputs`</b>: A 3D tensor.
* <b>`mask`</b>: Binary tensor of shape `(batch_size, steps)` indicating whether
  a given step should be masked (excluded from the average).


#### Input shape:

- If `data_format='channels_last'`:
  3D tensor with shape:
  `(batch_size, steps, features)`
- If `data_format='channels_first'`:
  3D tensor with shape:
  `(batch_size, features, steps)`



#### Output shape:

- If `keepdims`=False:
  2D tensor with shape `(batch_size, features)`.
- If `keepdims`=True:
  - If `data_format='channels_last'`:
    3D tensor with shape `(batch_size, 1, features)`
  - If `data_format='channels_first'`:
    3D tensor with shape `(batch_size, features, 1)`


