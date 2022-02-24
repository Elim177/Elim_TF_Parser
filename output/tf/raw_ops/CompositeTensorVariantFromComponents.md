description: Encodes an ExtensionType value into a variant scalar Tensor.
robots: noindex

# tf.raw_ops.CompositeTensorVariantFromComponents

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Encodes an `ExtensionType` value into a `variant` scalar Tensor.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.CompositeTensorVariantFromComponents`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.CompositeTensorVariantFromComponents(
    components, metadata, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Returns a scalar variant tensor containing a single `CompositeTensorVariant`
with the specified Tensor components and TypeSpec.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`components`
</td>
<td>
A list of `Tensor` objects.
The component tensors for the extension type value.
</td>
</tr><tr>
<td>
`metadata`
</td>
<td>
A `string`.
String serialization for the TypeSpec.  (Note: the encoding for the TypeSpec
may change in future versions of TensorFlow.)
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` of type `variant`.
</td>
</tr>

</table>

