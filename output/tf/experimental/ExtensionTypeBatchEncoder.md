description: Class used to encode and decode extension type values for batching.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.ExtensionTypeBatchEncoder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch"/>
<meta itemprop="property" content="decode"/>
<meta itemprop="property" content="encode"/>
<meta itemprop="property" content="encoding_specs"/>
<meta itemprop="property" content="unbatch"/>
</div>

# tf.experimental.ExtensionTypeBatchEncoder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/extension_type.py">View source</a>



Class used to encode and decode extension type values for batching.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.experimental.ExtensionTypeBatchEncoder`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

In order to be batched and unbatched by APIs such as <a href="../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a>,
<a href="../../tf/keras.md"><code>tf.keras</code></a>, and <a href="../../tf/map_fn.md"><code>tf.map_fn</code></a>, extension type values must be encoded as a list
of <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a>s, where stacking, unstacking, or concatenating these encoded
tensors and then decoding the result must be equivalent to stacking,
unstacking, or concatenating the original values. `ExtensionTypeBatchEncoder`s
are responsible for implementing this encoding.

The default `ExtensionTypeBatchEncoder` that is used by
`BatchableExtensionType` assumes that extension type values can be stacked,
unstacked, or concatenated by simply stacking, unstacking, or concatenating
every nested `Tensor`, `ExtensionType`, `CompositeTensor`, and `TensorShape`
field.

Extension types where this is not the case will need to override
`__batch_encoder__` with a custom encoder that overrides the `batch`,
`unbatch`, `encode`, and `decode` methods. E.g.:

```
>>> class CustomBatchEncoder(ExtensionTypeBatchEncoder):
...   pass # Override batch(), unbatch(), encode(), and decode().
```

```
>>> class CustomType(BatchableExtensionType):
...   x: tf.Tensor
...   y: tf.Tensor
...   shape: tf.TensorShape
...   __batch_encoder__ = CustomBatchEncoder()
```

For example, <a href="../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a> and <a href="../../tf/sparse/SparseTensor.md"><code>tf.SparseTensor</code></a> both use custom batch
encodings which define ops to "box" and "unbox" individual values into
<a href="../../tf.md#variant"><code>tf.variant</code></a> tensors.

## Methods

<h3 id="batch"><code>batch</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/extension_type.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>batch(
    spec, batch_size
)
</code></pre>

Returns the TypeSpec representing a batch of values described by `spec`.

The default definition returns a `TypeSpec` that is equal to `spec`, except
that an outer axis with size `batch_size` is added to every nested
`TypeSpec` and `TensorShape` field.  Subclasses may override this default
definition, when necessary.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`spec`
</td>
<td>
The `TypeSpec` for an individual value.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
An `int` indicating the number of values that are batched
together, or `None` if the batch size is not known.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `TypeSpec` for a batch of values.
</td>
</tr>

</table>



<h3 id="decode"><code>decode</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/extension_type.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>decode(
    spec, encoded_value
)
</code></pre>

Decodes `value` from a batchable tensor encoding.

See `encode` for a description of the default encoding.  Subclasses may
override this default definition, when necessary.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`spec`
</td>
<td>
The TypeSpec for the result value.  If encoded values with spec `s`
were batched, then `spec` should be `s.batch(batch_size)`; or if encoded
values with spec `s` were unbatched, then `spec` should be
`s.unbatch()`.
</td>
</tr><tr>
<td>
`encoded_value`
</td>
<td>
A nest of values returned by `encode`; or a nest of
values that was formed by stacking, unstacking, or concatenating the
corresponding elements of values returned by `encode`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A value compatible with `type_spec`.
</td>
</tr>

</table>



<h3 id="encode"><code>encode</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/extension_type.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>encode(
    spec, value, minimum_rank=0
)
</code></pre>

Encodes `value` as a nest of batchable Tensors or CompositeTensors.

The default definition returns a flat tuple of all the `Tensor`s,
`CompositeTensor`s, and `ExtensionType`s from a depth-first traversal of
`value`'s fields. Subclasses may override this default definition, when
necessary.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`spec`
</td>
<td>
The TypeSpec of the value to encode.
</td>
</tr><tr>
<td>
`value`
</td>
<td>
A value compatible with `spec`.
</td>
</tr><tr>
<td>
`minimum_rank`
</td>
<td>
The minimum rank for the returned Tensors, CompositeTensors,
and ExtensionType values.  This can be used to ensure that the encoded
values can be unbatched this number of times.   If `minimum_rank>0`,
then `t.shape[:minimum_rank]` must be compatible for all values `t`
returned by `encode`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A nest (as defined by <a href="../../tf/nest.md"><code>tf.nest</code></a>) of <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a>s, batchable
`tf.CompositeTensor`s, or `tf.ExtensionType`s.  Stacking, unstacking, or
concatenating these encoded values and then decoding the result must be
equivalent to stacking, unstacking, or concatenating the original values.
</td>
</tr>

</table>



<h3 id="encoding_specs"><code>encoding_specs</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/extension_type.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>encoding_specs(
    spec
)
</code></pre>

Returns a list of `TensorSpec`(s) describing the encoding for `spec`.

See `encode` for a description of the default encoding.  Subclasses may
override this default definition, when necessary.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`spec`
</td>
<td>
The TypeSpec whose encoding should be described.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A nest (as defined by `tf.nest) of `tf.TypeSpec`, describing the values
that are returned by `self.encode(spec, ...)`.  All TypeSpecs in this
nest must be batchable.
</td>
</tr>

</table>



<h3 id="unbatch"><code>unbatch</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/extension_type.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>unbatch(
    spec
)
</code></pre>

Returns the TypeSpec for a single unbatched element in `spec`.

The default definition returns a `TypeSpec` that is equal to `spec`, except
that the outermost axis is removed from every nested `TypeSpec`, and
`TensorShape` field.  Subclasses may override this default definition, when
necessary.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`spec`
</td>
<td>
The `TypeSpec` for a batch of values.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `TypeSpec` for an individual value.
</td>
</tr>

</table>





