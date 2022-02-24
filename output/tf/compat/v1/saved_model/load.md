description: Loads the model from a SavedModel as specified by tags. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.saved_model.load" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.saved_model.load

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/saved_model/loader_impl.py">View source</a>



Loads the model from a SavedModel as specified by tags. (deprecated)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.saved_model.loader.load`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.saved_model.load(
    sess, tags, export_dir, import_scope=None, **saver_kwargs
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

<a href="../../../../tf/compat/v1/saved_model/load.md"><code>tf.compat.v1.saved_model.load</code></a> or <a href="../../../../tf/compat/v1/saved_model/load.md"><code>tf.compat.v1.saved_model.loader.load</code></a> is
not compatible with eager execution. Please use <a href="../../../../tf/saved_model/load.md"><code>tf.saved_model.load</code></a> instead
to load your model. You can refer to the [SavedModel guide]
(https://www.tensorflow.org/guide/saved_model) for more information as well as
"Importing SavedModels from TensorFlow 1.x" in the [`tf.saved_model.load`]
(https://www.tensorflow.org/api_docs/python/tf/saved_model/load) docstring.

#### How to Map Arguments

| TF1 Arg Name          | TF2 Arg Name    | Note                       |
| :-------------------- | :-------------- | :------------------------- |
| `sess`                | Not supported   | -                          |
| `tags`                | `tags`          | -                          |
| `export_dir`          | `export_dir`    | -                          |
| `import_scope`        | Not supported   | Name scopes are not needed.
:                       :                 : By default, variables are  :
:                       :                 : associated with the loaded :
:                       :                 : object and function names  :
:                       :                 : are deduped.               :
| `saver_kwargs`        | Not supported   | -                          |

#### Before & After Usage Example

Before:

```
with tf.compat.v1.Session(graph=tf.Graph()) as sess:
  tf.compat.v1.saved_model.loader.load(sess, ["foo-tag"], export_dir)
```

After:

```
model = tf.saved_model.load(export_dir, tags=["foo-tag"])
```


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.loader.load or tf.compat.v1.saved_model.load. There will be a new function for importing SavedModels in Tensorflow 2.0.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sess`
</td>
<td>
The TensorFlow session to restore the variables.
</td>
</tr><tr>
<td>
`tags`
</td>
<td>
Set of string tags to identify the required MetaGraphDef. These should
correspond to the tags used when saving the variables using the
SavedModel `save()` API.
</td>
</tr><tr>
<td>
`export_dir`
</td>
<td>
Directory in which the SavedModel protocol buffer and variables
to be loaded are located.
</td>
</tr><tr>
<td>
`import_scope`
</td>
<td>
Optional `string` -- if specified, prepend this string
followed by '/' to all loaded tensor names. This scope is applied to
tensor instances loaded into the passed session, but it is *not* written
through to the static `MetaGraphDef` protocol buffer that is returned.
</td>
</tr><tr>
<td>
`**saver_kwargs`
</td>
<td>
Optional keyword arguments passed through to Saver.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The `MetaGraphDef` protocol buffer loaded in the provided session. This
can be used to further extract signature-defs, collection-defs, etc.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`RuntimeError`
</td>
<td>
MetaGraphDef associated with the tags cannot be found.
</td>
</tr>
</table>


