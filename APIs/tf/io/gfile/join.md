description: Join one or more path components intelligently.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.io.gfile.join" />
<meta itemprop="path" content="Stable" />
</div>

# tf.io.gfile.join

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>



Join one or more path components intelligently.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.io.gfile.join`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.io.gfile.join(
    path, *paths
)
</code></pre>



<!-- Placeholder for "Used in" -->

TensorFlow specific filesystems will be joined
like a url (using "/" as the path seperator) on all platforms:

On Windows or Linux/Unix-like:
```
>>> tf.io.gfile.join("gcs://folder", "file.py")
'gcs://folder/file.py'
```

```
>>> tf.io.gfile.join("ram://folder", "file.py")
'ram://folder/file.py'
```

But the native filesystem is handled just like os.path.join:

```
>>> path = tf.io.gfile.join("folder", "file.py")
>>> if os.name == "nt":
...   expected = "folder\\file.py"  # Windows
... else:
...   expected = "folder/file.py"  # Linux/Unix-like
>>> path == expected
True
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`path`
</td>
<td>
string, path to a directory
</td>
</tr><tr>
<td>
`paths`
</td>
<td>
string, additional paths to concatenate
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`path`
</td>
<td>
the joined path.
</td>
</tr>
</table>

