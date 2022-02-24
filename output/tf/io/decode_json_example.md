description: Convert JSON-encoded Example records to binary protocol buffer strings.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.io.decode_json_example" />
<meta itemprop="path" content="Stable" />
</div>

# tf.io.decode_json_example

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/parsing_ops.py">View source</a>



Convert JSON-encoded Example records to binary protocol buffer strings.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.decode_json_example`, `tf.compat.v1.io.decode_json_example`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.io.decode_json_example(
    json_examples, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Note: This is **not** a general purpose JSON parsing op.

This op converts JSON-serialized <a href="../../tf/train/Example.md"><code>tf.train.Example</code></a> (maybe created with
`json_format.MessageToJson`, following the
[standard JSON mapping](
https://developers.google.com/protocol-buffers/docs/proto3#json))
to a binary-serialized <a href="../../tf/train/Example.md"><code>tf.train.Example</code></a> (equivalent to
<a href="../../tf/train/BytesList.md#SerializeToString"><code>Example.SerializeToString()</code></a>) suitable for conversion to tensors with
<a href="../../tf/io/parse_example.md"><code>tf.io.parse_example</code></a>.

Here is a <a href="../../tf/train/Example.md"><code>tf.train.Example</code></a> proto:

```
>>> example = tf.train.Example(
...   features=tf.train.Features(
...       feature={
...           "a": tf.train.Feature(
...               int64_list=tf.train.Int64List(
...                   value=[1, 1, 3]))}))
```

Here it is converted to JSON:

```
>>> from google.protobuf import json_format
>>> example_json = json_format.MessageToJson(example)
>>> print(example_json)
{
  "features": {
    "feature": {
      "a": {
        "int64List": {
          "value": [
            "1",
            "1",
            "3"
          ]
        }
      }
    }
  }
}
```

This op converts the above json string to a binary proto:

```
>>> example_binary = tf.io.decode_json_example(example_json)
>>> example_binary.numpy()
b'\n\x0f\n\r\n\x01a\x12\x08\x1a\x06\x08\x01\x08\x01\x08\x03'
```

The OP works on string tensors of andy shape:

```
>>> tf.io.decode_json_example([
...     [example_json, example_json],
...     [example_json, example_json]]).shape.as_list()
[2, 2]
```

This resulting binary-string is equivalent to <a href="../../tf/train/BytesList.md#SerializeToString"><code>Example.SerializeToString()</code></a>,
and can be converted to Tensors using <a href="../../tf/io/parse_example.md"><code>tf.io.parse_example</code></a> and related
functions:

```
>>> tf.io.parse_example(
...   serialized=[example_binary.numpy(),
...              example.SerializeToString()],
...   features = {'a': tf.io.FixedLenFeature(shape=[3], dtype=tf.int64)})
{'a': <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
 array([[1, 1, 3],
        [1, 1, 3]])>}
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`json_examples`
</td>
<td>
A string tensor containing json-serialized `tf.Example`
protos.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the op.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A string Tensor containing the binary-serialized `tf.Example` protos.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>
<tr class="alt">
<td colspan="2">
<a href="../../tf/errors/InvalidArgumentError.md"><code>tf.errors.InvalidArgumentError</code></a>: If the JSON could not be converted to a
`tf.Example`
</td>
</tr>

</table>

