description: A preprocessing layer which maps text features to integer sequences.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.TextVectorization" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="adapt"/>
<meta itemprop="property" content="compile"/>
<meta itemprop="property" content="get_vocabulary"/>
<meta itemprop="property" content="reset_state"/>
<meta itemprop="property" content="set_vocabulary"/>
<meta itemprop="property" content="update_state"/>
<meta itemprop="property" content="vocabulary_size"/>
</div>

# tf.keras.layers.TextVectorization

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/preprocessing/text_vectorization.py#L45-L546">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A preprocessing layer which maps text features to integer sequences.

Inherits From: [`PreprocessingLayer`](../../../tf/keras/layers/experimental/preprocessing/PreprocessingLayer.md), [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.experimental.preprocessing.TextVectorization`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.TextVectorization(
    max_tokens=None, standardize=&#x27;lower_and_strip_punctuation&#x27;,
    split=&#x27;whitespace&#x27;, ngrams=None, output_mode=&#x27;int&#x27;,
    output_sequence_length=None, pad_to_max_tokens=(False), vocabulary=None,
    idf_weights=None, sparse=(False), ragged=(False), **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer has basic options for managing text in a Keras model. It transforms
a batch of strings (one example = one string) into either a list of token
indices (one example = 1D tensor of integer token indices) or a dense
representation (one example = 1D tensor of float values representing data
about the example's tokens).

The vocabulary for the layer must be either supplied on construction or
learned via `adapt()`. When this layer is adapted, it will analyze the
dataset, determine the frequency of individual string values, and create a
vocabulary from them. This vocabulary can have unlimited size or be capped,
depending on the configuration options for this layer; if there are more
unique values in the input than the maximum vocabulary size, the most frequent
terms will be used to create the vocabulary.

The processing of each example contains the following steps:

1. Standardize each example (usually lowercasing + punctuation stripping)
2. Split each example into substrings (usually words)
3. Recombine substrings into tokens (usually ngrams)
4. Index tokens (associate a unique int value with each token)
5. Transform each example using this index, either into a vector of ints or
   a dense float vector.

Some notes on passing callables to customize splitting and normalization for
this layer:

1. Any callable can be passed to this Layer, but if you want to serialize
   this object you should only pass functions that are registered Keras
   serializables (see <a href="../../../tf/keras/utils/register_keras_serializable.md"><code>tf.keras.utils.register_keras_serializable</code></a> for more
   details).
2. When using a custom callable for `standardize`, the data received
   by the callable will be exactly as passed to this layer. The callable
   should return a tensor of the same shape as the input.
3. When using a custom callable for `split`, the data received by the
   callable will have the 1st dimension squeezed out - instead of
   `[["string to split"], ["another string to split"]]`, the Callable will
   see `["string to split", "another string to split"]`. The callable should
   return a Tensor with the first dimension containing the split tokens -
   in this example, we should see something like `[["string", "to",
   "split"], ["another", "string", "to", "split"]]`. This makes the callable
   site natively compatible with <a href="../../../tf/strings/split.md"><code>tf.strings.split()</code></a>.

For an overview and full list of preprocessing layers, see the preprocessing
[guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`max_tokens`
</td>
<td>
Maximum size of the vocabulary for this layer. This should only
be specified when adapting a vocabulary or when setting
`pad_to_max_tokens=True`. Note that this vocabulary
contains 1 OOV token, so the effective number of tokens is `(max_tokens -
1 - (1 if output_mode == "int" else 0))`.
</td>
</tr><tr>
<td>
`standardize`
</td>
<td>
Optional specification for standardization to apply to the
input text. Values can be None (no standardization),
`"lower_and_strip_punctuation"` (lowercase and remove punctuation) or a
Callable. Default is `"lower_and_strip_punctuation"`.
</td>
</tr><tr>
<td>
`split`
</td>
<td>
Optional specification for splitting the input text. Values can be
None (no splitting), `"whitespace"` (split on ASCII whitespace), or a
Callable. The default is `"whitespace"`.
</td>
</tr><tr>
<td>
`ngrams`
</td>
<td>
Optional specification for ngrams to create from the possibly-split
input text. Values can be None, an integer or tuple of integers; passing
an integer will create ngrams up to that integer, and passing a tuple of
integers will create ngrams for the specified values in the tuple. Passing
None means that no ngrams will be created.
</td>
</tr><tr>
<td>
`output_mode`
</td>
<td>
Optional specification for the output of the layer. Values can
be `"int"`, `"multi_hot"`, `"count"` or `"tf_idf"`, configuring the layer
as follows:
  - `"int"`: Outputs integer indices, one integer index per split string
    token. When `output_mode == "int"`, 0 is reserved for masked
    locations; this reduces the vocab size to
    `max_tokens - 2` instead of `max_tokens - 1`.
  - `"multi_hot"`: Outputs a single int array per batch, of either
    vocab_size or max_tokens size, containing 1s in all elements where the
    token mapped to that index exists at least once in the batch item.
  - `"count"`: Like `"multi_hot"`, but the int array contains a count of
    the number of times the token at that index appeared in the
    batch item.
  - `"tf_idf"`: Like `"multi_hot"`, but the TF-IDF algorithm is applied to
    find the value in each token slot.
For `"int"` output, any shape of input and output is supported. For all
other output modes, currently only rank 1 inputs (and rank 2 outputs after
splitting) are supported.
</td>
</tr><tr>
<td>
`output_sequence_length`
</td>
<td>
Only valid in INT mode. If set, the output will have
its time dimension padded or truncated to exactly `output_sequence_length`
values, resulting in a tensor of shape
`(batch_size, output_sequence_length)` regardless of how many tokens
resulted from the splitting step. Defaults to None.
</td>
</tr><tr>
<td>
`pad_to_max_tokens`
</td>
<td>
Only valid in  `"multi_hot"`, `"count"`, and `"tf_idf"`
modes. If True, the output will have its feature axis padded to
`max_tokens` even if the number of unique tokens in the vocabulary is less
than max_tokens, resulting in a tensor of shape `(batch_size, max_tokens)`
regardless of vocabulary size. Defaults to False.
</td>
</tr><tr>
<td>
`vocabulary`
</td>
<td>
Optional. Either an array of strings or a string path to a text
file. If passing an array, can pass a tuple, list, 1D numpy array, or 1D
tensor containing the string vocbulary terms. If passing a file path, the
file should contain one line per term in the vocabulary. If this argument
is set, there is no need to `adapt` the layer.
</td>
</tr><tr>
<td>
`idf_weights`
</td>
<td>
Only valid when `output_mode` is `"tf_idf"`. A tuple, list, 1D
numpy array, or 1D tensor or the same length as the vocabulary, containing
the floating point inverse document frequency weights, which will be
multiplied by per sample term counts for the final `tf_idf` weight. If the
`vocabulary` argument is set, and `output_mode` is `"tf_idf"`, this
argument must be supplied.
</td>
</tr><tr>
<td>
`ragged`
</td>
<td>
Boolean. Only applicable to `"int"` output mode. If True, returns a
`RaggedTensor` instead of a dense `Tensor`, where each sequence may have a
different length after string splitting. Defaults to False.
</td>
</tr><tr>
<td>
`sparse`
</td>
<td>
Boolean. Only applicable to `"multi_hot"`, `"count"`, and
`"tf_idf"` output modes. If True, returns a `SparseTensor` instead of a
dense `Tensor`. Defaults to False.
</td>
</tr>
</table>



#### Example:



This example instantiates a `TextVectorization` layer that lowercases text,
splits on whitespace, strips punctuation, and outputs integer vocab indices.

```
>>> text_dataset = tf.data.Dataset.from_tensor_slices(["foo", "bar", "baz"])
>>> max_features = 5000  # Maximum vocab size.
>>> max_len = 4  # Sequence length to pad the outputs to.
>>>
>>> # Create the layer.
>>> vectorize_layer = tf.keras.layers.TextVectorization(
...  max_tokens=max_features,
...  output_mode='int',
...  output_sequence_length=max_len)
>>>
>>> # Now that the vocab layer has been created, call `adapt` on the text-only
>>> # dataset to create the vocabulary. You don't have to batch, but for large
>>> # datasets this means we're not keeping spare copies of the dataset.
>>> vectorize_layer.adapt(text_dataset.batch(64))
>>>
>>> # Create the model that uses the vectorize text layer
>>> model = tf.keras.models.Sequential()
>>>
>>> # Start by creating an explicit input layer. It needs to have a shape of
>>> # (1,) (because we need to guarantee that there is exactly one string
>>> # input per batch), and the dtype needs to be 'string'.
>>> model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
>>>
>>> # The first layer in our model is the vectorization layer. After this
>>> # layer, we have a tensor of shape (batch_size, max_len) containing vocab
>>> # indices.
>>> model.add(vectorize_layer)
>>>
>>> # Now, the model can map strings to integers, and you can add an embedding
>>> # layer to map these integers to learned embeddings.
>>> input_data = [["foo qux bar"], ["qux baz"]]
>>> model.predict(input_data)
array([[2, 1, 4, 0],
       [1, 3, 0, 0]])
```

#### Example:



This example instantiates a `TextVectorization` layer by passing a list
of vocabulary terms to the layer's `__init__()` method.

```
>>> vocab_data = ["earth", "wind", "and", "fire"]
>>> max_len = 4  # Sequence length to pad the outputs to.
>>>
>>> # Create the layer, passing the vocab directly. You can also pass the
>>> # vocabulary arg a path to a file containing one vocabulary word per
>>> # line.
>>> vectorize_layer = tf.keras.layers.TextVectorization(
...  max_tokens=max_features,
...  output_mode='int',
...  output_sequence_length=max_len,
...  vocabulary=vocab_data)
>>>
>>> # Because we've passed the vocabulary directly, we don't need to adapt
>>> # the layer - the vocabulary is already set. The vocabulary contains the
>>> # padding token ('') and OOV token ('[UNK]') as well as the passed tokens.
>>> vectorize_layer.get_vocabulary()
['', '[UNK]', 'earth', 'wind', 'and', 'fire']
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`is_adapted`
</td>
<td>
Whether the layer has been fit to data already.
</td>
</tr>
</table>



## Methods

<h3 id="adapt"><code>adapt</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/engine/base_preprocessing_layer.py#L156-L248">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>adapt(
    data, batch_size=None, steps=None
)
</code></pre>

Fits the state of the preprocessing layer to the data being passed.

After calling `adapt` on a layer, a preprocessing layer's state will not
update during training. In order to make preprocessing layers efficient in
any distribution context, they are kept constant with respect to any
compiled <a href="../../../tf/Graph.md"><code>tf.Graph</code></a>s that call the layer. This does not affect the layer use
when adapting each layer only once, but if you adapt a layer multiple times
you will need to take care to re-compile any compiled functions as follows:

 * If you are adding a preprocessing layer to a <a href="../../../tf/keras/Model.md"><code>keras.Model</code></a>, you need to
   call `model.compile` after each subsequent call to `adapt`.
 * If you are calling a preprocessing layer inside <a href="../../../tf/data/Dataset.md#map"><code>tf.data.Dataset.map</code></a>,
   you should call `map` again on the input <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> after each
   `adapt`.
 * If you are using a <a href="../../../tf/function.md"><code>tf.function</code></a> directly which calls a preprocessing
   layer, you need to call <a href="../../../tf/function.md"><code>tf.function</code></a> again on your callable after
   each subsequent call to `adapt`.

<a href="../../../tf/keras/Model.md"><code>tf.keras.Model</code></a> example with multiple adapts:

```
>>> layer = tf.keras.layers.experimental.preprocessing.Normalization(
...     axis=None)
>>> layer.adapt([0, 2])
>>> model = tf.keras.Sequential(layer)
>>> model.predict([0, 1, 2])
array([-1.,  0.,  1.], dtype=float32)
>>> layer.adapt([-1, 1])
>>> model.compile() # This is needed to re-compile model.predict!
>>> model.predict([0, 1, 2])
array([0., 1., 2.], dtype=float32)
```

<a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> example with multiple adapts:

```
>>> layer = tf.keras.layers.experimental.preprocessing.Normalization(
...     axis=None)
>>> layer.adapt([0, 2])
>>> input_ds = tf.data.Dataset.range(3)
>>> normalized_ds = input_ds.map(layer)
>>> list(normalized_ds.as_numpy_iterator())
[array([-1.], dtype=float32),
 array([0.], dtype=float32),
 array([1.], dtype=float32)]
>>> layer.adapt([-1, 1])
>>> normalized_ds = input_ds.map(layer) # Re-map over the input dataset.
>>> list(normalized_ds.as_numpy_iterator())
[array([0.], dtype=float32),
 array([1.], dtype=float32),
 array([2.], dtype=float32)]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`data`
</td>
<td>
The data to train on. It can be passed either as a tf.data
Dataset, or as a numpy array.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
Integer or `None`.
Number of samples per state update.
If unspecified, `batch_size` will default to 32.
Do not specify the `batch_size` if your data is in the
form of datasets, generators, or <a href="../../../tf/keras/utils/Sequence.md"><code>keras.utils.Sequence</code></a> instances
(since they generate batches).
</td>
</tr><tr>
<td>
`steps`
</td>
<td>
Integer or `None`.
Total number of steps (batches of samples)
When training with input tensors such as
TensorFlow data tensors, the default `None` is equal to
the number of samples in your dataset divided by
the batch size, or 1 if that cannot be determined. If x is a
<a href="../../../tf/data.md"><code>tf.data</code></a> dataset, and 'steps' is None, the epoch will run until
the input dataset is exhausted. When passing an infinitely
repeating dataset, you must specify the `steps` argument. This
argument is not supported with array inputs.
</td>
</tr>
</table>



<h3 id="compile"><code>compile</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/engine/base_preprocessing_layer.py#L134-L154">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compile(
    run_eagerly=None, steps_per_execution=None
)
</code></pre>

Configures the layer for `adapt`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`run_eagerly`
</td>
<td>
Bool. Defaults to `False`. If `True`, this `Model`'s logic
will not be wrapped in a <a href="../../../tf/function.md"><code>tf.function</code></a>. Recommended to leave this as
`None` unless your `Model` cannot be run inside a <a href="../../../tf/function.md"><code>tf.function</code></a>.
steps_per_execution: Int. Defaults to 1. The number of batches to run
  during each <a href="../../../tf/function.md"><code>tf.function</code></a> call. Running multiple batches inside a
  single <a href="../../../tf/function.md"><code>tf.function</code></a> call can greatly improve performance on TPUs or
  small models with a large Python overhead.
</td>
</tr>
</table>



<h3 id="get_vocabulary"><code>get_vocabulary</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/preprocessing/text_vectorization.py#L373-L382">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_vocabulary(
    include_special_tokens=(True)
)
</code></pre>

Returns the current vocabulary of the layer.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`include_special_tokens`
</td>
<td>
If True, the returned vocabulary will include
the padding and OOV tokens, and a term's index in the vocabulary will
equal the term's index when calling the layer. If False, the returned
vocabulary will not include any padding or OOV tokens.
</td>
</tr>
</table>



<h3 id="reset_state"><code>reset_state</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/preprocessing/text_vectorization.py#L370-L371">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_state()
</code></pre>

Resets the statistics of the preprocessing layer.


<h3 id="set_vocabulary"><code>set_vocabulary</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/preprocessing/text_vectorization.py#L411-L437">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_vocabulary(
    vocabulary, idf_weights=None
)
</code></pre>

Sets vocabulary (and optionally document frequency) data for this layer.

This method sets the vocabulary and idf weights for this layer directly,
instead of analyzing a dataset through 'adapt'. It should be used whenever
the vocab (and optionally document frequency) information is already known.
If vocabulary data is already present in the layer, this method will replace
it.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`vocabulary`
</td>
<td>
Either an array or a string path to a text file. If passing an
array, can pass a tuple, list, 1D numpy array, or 1D tensor containing
the vocbulary terms. If passing a file path, the file should contain one
line per term in the vocabulary.
</td>
</tr><tr>
<td>
`idf_weights`
</td>
<td>
A tuple, list, 1D numpy array, or 1D tensor of inverse
document frequency weights with equal length to vocabulary. Must be set
if `output_mode` is `"tf_idf"`. Should not be set otherwise.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If there are too many inputs, the inputs do not match, or
input data is missing.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
If the vocabulary cannot be set when this function is
called. This happens when `"multi_hot"`, `"count"`, and "tf_idf" modes,
if `pad_to_max_tokens` is False and the layer itself has already been
called.
</td>
</tr>
</table>



<h3 id="update_state"><code>update_state</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/preprocessing/text_vectorization.py#L364-L365">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_state(
    data
)
</code></pre>

Accumulates statistics for the preprocessing layer.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`data`
</td>
<td>
A mini-batch of inputs to the layer.
</td>
</tr>
</table>



<h3 id="vocabulary_size"><code>vocabulary_size</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/preprocessing/text_vectorization.py#L384-L390">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>vocabulary_size()
</code></pre>

Gets the current size of the layer's vocabulary.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The integer size of the voculary, including optional mask and oov indices.
</td>
</tr>

</table>





