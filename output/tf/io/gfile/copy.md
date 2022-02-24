description: Copies data from src to dst.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.io.gfile.copy" />
<meta itemprop="path" content="Stable" />
</div>

# tf.io.gfile.copy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>



Copies data from `src` to `dst`.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.io.gfile.copy`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.io.gfile.copy(
    src, dst, overwrite=(False)
)
</code></pre>



<!-- Placeholder for "Used in" -->

```
>>> with open("/tmp/x", "w") as f:
...   f.write("asdf")
...
4
>>> tf.io.gfile.exists("/tmp/x")
True
>>> tf.io.gfile.copy("/tmp/x", "/tmp/y")
>>> tf.io.gfile.exists("/tmp/y")
True
>>> tf.io.gfile.remove("/tmp/y")
```

You can also specify the URI scheme for selecting a different filesystem:

```
>>> with open("/tmp/x", "w") as f:
...   f.write("asdf")
...
4
>>> tf.io.gfile.copy("/tmp/x", "file:///tmp/y")
>>> tf.io.gfile.exists("/tmp/y")
True
>>> tf.io.gfile.remove("/tmp/y")
```

Note that you need to always specify a file name, even if moving into a new
directory. This is because some cloud filesystems don't have the concept of a
directory.

```
>>> with open("/tmp/x", "w") as f:
...   f.write("asdf")
...
4
>>> tf.io.gfile.mkdir("/tmp/new_dir")
>>> tf.io.gfile.copy("/tmp/x", "/tmp/new_dir/y")
>>> tf.io.gfile.exists("/tmp/new_dir/y")
True
>>> tf.io.gfile.rmtree("/tmp/new_dir")
```

If you want to prevent errors if the path already exists, you can use
`overwrite` argument:

```
>>> with open("/tmp/x", "w") as f:
...   f.write("asdf")
...
4
>>> tf.io.gfile.copy("/tmp/x", "file:///tmp/y")
>>> tf.io.gfile.copy("/tmp/x", "file:///tmp/y", overwrite=True)
>>> tf.io.gfile.remove("/tmp/y")
```

Note that the above will still result in an error if you try to overwrite a
directory with a file.

Note that you cannot copy a directory, only file arguments are supported.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`src`
</td>
<td>
string, name of the file whose contents need to be copied
</td>
</tr><tr>
<td>
`dst`
</td>
<td>
string, name of the file to which to copy to
</td>
</tr><tr>
<td>
`overwrite`
</td>
<td>
boolean, if false it's an error for `dst` to be occupied by an
existing file.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`errors.OpError`
</td>
<td>
If the operation fails.
</td>
</tr>
</table>

