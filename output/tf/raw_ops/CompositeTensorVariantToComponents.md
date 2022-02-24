description: Decodes a variant scalar Tensor into an ExtensionType value.
robots: noindex

# tf.raw_ops.CompositeTensorVariantToComponents

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Decodes a `variant` scalar Tensor into an `ExtensionType` value.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.CompositeTensorVariantToComponents`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.CompositeTensorVariantToComponents(
    encoded, metadata, Tcomponents, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Returns the Tensor components encoded in a `CompositeTensorVariant`.

Raises an error if `type_spec_proto` doesn't match the TypeSpec
in `encoded`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`encoded`
</td>
<td>
A `Tensor` of type `variant`.
A scalar `variant` Tensor containing an encoded ExtensionType value.
</td>
</tr><tr>
<td>
`metadata`
</td>
<td>
A `string`.
String serialization for the TypeSpec.  Must be compatible with the
`TypeSpec` contained in `encoded`.  (Note: the encoding for the TypeSpec
may change in future versions of TensorFlow.)
</td>
</tr><tr>
<td>
`Tcomponents`
</td>
<td>
A list of `tf.DTypes`. Expected dtypes for components.
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
A list of `Tensor` objects of type `Tcomponents`.
</td>
</tr>

</table>

