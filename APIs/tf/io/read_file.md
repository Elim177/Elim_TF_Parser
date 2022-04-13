description: Reads the contents of file.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.io.read_file" />
<meta itemprop="path" content="Stable" />
</div>

# tf.io.read_file

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/io_ops.py">View source</a>



Reads the contents of file.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.io.read_file`, `tf.compat.v1.read_file`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.io.read_file(
    filename, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This operation returns a tensor with the entire contents of the input
filename. It does not do any parsing, it just returns the contents as
they are. Usually, this is the first step in the input pipeline.

#### Example:



```
>>> with open("/tmp/file.txt", "w") as f:
...   f.write("asdf")
...
4
>>> tf.io.read_file("/tmp/file.txt")
<tf.Tensor: shape=(), dtype=string, numpy=b'asdf'>
```

Example of using the op in a function to read an image, decode it and reshape
the tensor containing the pixel data:

```
>>> @tf.function
... def load_image(filename):
...   raw = tf.io.read_file(filename)
...   image = tf.image.decode_png(raw, channels=3)
...   # the `print` executes during tracing.
...   print("Initial shape: ", image.shape)
...   image.set_shape([28, 28, 3])
...   print("Final shape: ", image.shape)
...   return image
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`filename`
</td>
<td>
string. filename to read from.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
string.  Optional name for the op.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor of dtype "string", with the file contents.
</td>
</tr>

</table>

