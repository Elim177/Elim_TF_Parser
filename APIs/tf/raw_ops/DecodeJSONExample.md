description: Convert JSON-encoded Example records to binary protocol buffer strings.
robots: noindex

# tf.raw_ops.DecodeJSONExample

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Convert JSON-encoded Example records to binary protocol buffer strings.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.DecodeJSONExample`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.DecodeJSONExample(
    json_examples, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


Note: This is **not** a general purpose JSON parsing op.

This op converts JSON-serialized
<a href="../../tf/train/Example.md"><code>tf.train.Example</code></a> (created with `json_format.MessageToJson`, following the
[standard JSON mapping](https://developers.google.com/protocol-buffers/docs/proto3#json))
to a binary-serialized <a href="../../tf/train/Example.md"><code>tf.train.Example</code></a> (equivalent to
<a href="../../tf/train/BytesList.md#SerializeToString"><code>Example.SerializeToString()</code></a>) suitable for conversion to tensors with
<a href="../../tf/io/parse_example.md"><code>tf.io.parse_example</code></a>.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`json_examples`
</td>
<td>
A `Tensor` of type `string`.
Each string is a JSON object serialized according to the JSON
mapping of the Example proto.
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
A `Tensor` of type `string`.
</td>
</tr>

</table>

