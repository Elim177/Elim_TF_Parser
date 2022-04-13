description: A preprocessing layer which hashes and bins categorical features.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.Hashing" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf.keras.layers.Hashing

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/preprocessing/hashing.py#L27-L206">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A preprocessing layer which hashes and bins categorical features.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.experimental.preprocessing.Hashing`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.layers.Hashing`, `tf.compat.v1.keras.layers.experimental.preprocessing.Hashing`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.Hashing(
    num_bins, mask_value=None, salt=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer transforms categorical inputs to hashed output. It element-wise
converts a ints or strings to ints in a fixed range. The stable hash
function uses `tensorflow::ops::Fingerprint` to produce the same output
consistently across all platforms.

This layer uses [FarmHash64](https://github.com/google/farmhash) by default,
which provides a consistent hashed output across different platforms and is
stable across invocations, regardless of device and context, by mixing the
input bits thoroughly.

If you want to obfuscate the hashed output, you can also pass a random `salt`
argument in the constructor. In that case, the layer will use the
[SipHash64](https://github.com/google/highwayhash) hash function, with
the `salt` value serving as additional input to the hash function.

For an overview and full list of preprocessing layers, see the preprocessing
[guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

**Example (FarmHash64)**

```
>>> layer = tf.keras.layers.Hashing(num_bins=3)
>>> inp = [['A'], ['B'], ['C'], ['D'], ['E']]
>>> layer(inp)
<tf.Tensor: shape=(5, 1), dtype=int64, numpy=
  array([[1],
         [0],
         [1],
         [1],
         [2]])>
```

**Example (FarmHash64) with a mask value**

```
>>> layer = tf.keras.layers.Hashing(num_bins=3, mask_value='')
>>> inp = [['A'], ['B'], [''], ['C'], ['D']]
>>> layer(inp)
<tf.Tensor: shape=(5, 1), dtype=int64, numpy=
  array([[1],
         [1],
         [0],
         [2],
         [2]])>
```

**Example (SipHash64)**

```
>>> layer = tf.keras.layers.Hashing(num_bins=3, salt=[133, 137])
>>> inp = [['A'], ['B'], ['C'], ['D'], ['E']]
>>> layer(inp)
<tf.Tensor: shape=(5, 1), dtype=int64, numpy=
  array([[1],
         [2],
         [1],
         [0],
         [2]])>
```

**Example (Siphash64 with a single integer, same as `salt=[133, 133]`)**

```
>>> layer = tf.keras.layers.Hashing(num_bins=3, salt=133)
>>> inp = [['A'], ['B'], ['C'], ['D'], ['E']]
>>> layer(inp)
<tf.Tensor: shape=(5, 1), dtype=int64, numpy=
  array([[0],
         [0],
         [2],
         [1],
         [0]])>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`num_bins`
</td>
<td>
Number of hash bins. Note that this includes the `mask_value` bin,
so the effective number of bins is `(num_bins - 1)` if `mask_value` is
set.
</td>
</tr><tr>
<td>
`mask_value`
</td>
<td>
A value that represents masked inputs, which are mapped to
index 0. Defaults to None, meaning no mask term will be added and the
hashing will start at index 0.
</td>
</tr><tr>
<td>
`salt`
</td>
<td>
A single unsigned integer or None.
If passed, the hash function used will be SipHash64, with these values
used as an additional input (known as a "salt" in cryptography).
These should be non-zero. Defaults to `None` (in that
case, the FarmHash64 hash function is used). It also supports
tuple/list of 2 unsigned integer numbers, see reference paper for details.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword arguments to construct a layer.
</td>
</tr>
</table>



#### Input shape:

A single or list of string, int32 or int64 `Tensor`,
`SparseTensor` or `RaggedTensor` of shape `(batch_size, ...,)`



#### Output shape:

An int64 `Tensor`, `SparseTensor` or `RaggedTensor` of shape
`(batch_size, ...)`. If any input is `RaggedTensor` then output is
`RaggedTensor`, otherwise if any input is `SparseTensor` then output is
`SparseTensor`, otherwise the output is `Tensor`.



#### Reference:

- [SipHash with salt](https://www.131002.net/siphash/siphash.pdf)


