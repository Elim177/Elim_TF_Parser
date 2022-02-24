description: A preprocessing layer which encodes integer features.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.CategoryEncoding" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf.keras.layers.CategoryEncoding

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/preprocessing/category_encoding.py#L35-L208">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A preprocessing layer which encodes integer features.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.experimental.preprocessing.CategoryEncoding`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.layers.CategoryEncoding`, `tf.compat.v1.keras.layers.experimental.preprocessing.CategoryEncoding`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.CategoryEncoding(
    num_tokens=None, output_mode=&#x27;multi_hot&#x27;, sparse=(False), **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer provides options for condensing data into a categorical encoding
when the total number of tokens are known in advance. It accepts integer
values as inputs, and it outputs a dense or sparse representation of those
inputs. For integer inputs where the total number of tokens is not known, use
<a href="../../../tf/keras/layers/IntegerLookup.md"><code>tf.keras.layers.IntegerLookup</code></a> instead.

For an overview and full list of preprocessing layers, see the preprocessing
[guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

#### Examples:



**One-hot encoding data**

```
>>> layer = tf.keras.layers.CategoryEncoding(
...           num_tokens=4, output_mode="one_hot")
>>> layer([3, 2, 0, 1])
<tf.Tensor: shape=(4, 4), dtype=float32, numpy=
  array([[0., 0., 0., 1.],
         [0., 0., 1., 0.],
         [1., 0., 0., 0.],
         [0., 1., 0., 0.]], dtype=float32)>
```

**Multi-hot encoding data**

```
>>> layer = tf.keras.layers.CategoryEncoding(
...           num_tokens=4, output_mode="multi_hot")
>>> layer([[0, 1], [0, 0], [1, 2], [3, 1]])
<tf.Tensor: shape=(4, 4), dtype=float32, numpy=
  array([[1., 1., 0., 0.],
         [1., 0., 0., 0.],
         [0., 1., 1., 0.],
         [0., 1., 0., 1.]], dtype=float32)>
```

**Using weighted inputs in `"count"` mode**

```
>>> layer = tf.keras.layers.CategoryEncoding(
...           num_tokens=4, output_mode="count")
>>> count_weights = np.array([[.1, .2], [.1, .1], [.2, .3], [.4, .2]])
>>> layer([[0, 1], [0, 0], [1, 2], [3, 1]], count_weights=count_weights)
<tf.Tensor: shape=(4, 4), dtype=float64, numpy=
  array([[0.1, 0.2, 0. , 0. ],
         [0.2, 0. , 0. , 0. ],
         [0. , 0.2, 0.3, 0. ],
         [0. , 0.2, 0. , 0.4]])>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`num_tokens`
</td>
<td>
The total number of tokens the layer should support. All inputs
to the layer must integers in the range `0 <= value < num_tokens`, or an
error will be thrown.
</td>
</tr><tr>
<td>
`output_mode`
</td>
<td>
Specification for the output of the layer.
Defaults to `"multi_hot"`. Values can be `"one_hot"`, `"multi_hot"` or
`"count"`, configuring the layer as follows:
  - `"one_hot"`: Encodes each individual element in the input into an
    array of `num_tokens` size, containing a 1 at the element index. If
    the last dimension is size 1, will encode on that dimension. If the
    last dimension is not size 1, will append a new dimension for the
    encoded output.
  - `"multi_hot"`: Encodes each sample in the input into a single array
    of `num_tokens` size, containing a 1 for each vocabulary term present
    in the sample. Treats the last dimension as the sample dimension, if
    input shape is `(..., sample_length)`, output shape will be
    `(..., num_tokens)`.
  - `"count"`: Like `"multi_hot"`, but the int array contains a count of
    the number of times the token at that index appeared in the sample.
For all output modes, currently only output up to rank 2 is supported.
</td>
</tr><tr>
<td>
`sparse`
</td>
<td>
Boolean. If true, returns a `SparseTensor` instead of a dense
`Tensor`. Defaults to `False`.
</td>
</tr>
</table>



#### Call arguments:


* <b>`inputs`</b>: A 1D or 2D tensor of integer inputs.
* <b>`count_weights`</b>: A tensor in the same shape as `inputs` indicating the
  weight for each sample value when summing up in `count` mode. Not used in
  `"multi_hot"` or `"one_hot"` modes.


