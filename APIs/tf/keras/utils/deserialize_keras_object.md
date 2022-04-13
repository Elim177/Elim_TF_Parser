description: Turns the serialized form of a Keras object back into an actual object.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.deserialize_keras_object" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.utils.deserialize_keras_object

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/utils/generic_utils.py#L609-L724">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Turns the serialized form of a Keras object back into an actual object.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.utils.deserialize_keras_object`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.deserialize_keras_object(
    identifier, module_objects=None, custom_objects=None,
    printable_module_name=&#x27;object&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

This function is for mid-level library implementers rather than end users.

Importantly, this utility requires you to provide the dict of `module_objects`
to use for looking up the object config; this is not populated by default.
If you need a deserialization utility that has preexisting knowledge of
built-in Keras objects, use e.g. <a href="../../../tf/keras/layers/deserialize.md"><code>keras.layers.deserialize(config)</code></a>,
<a href="../../../tf/keras/metrics/deserialize.md"><code>keras.metrics.deserialize(config)</code></a>, etc.

Calling `deserialize_keras_object` while underneath the
`SharedObjectLoadingScope` context manager will cause any already-seen shared
objects to be returned as-is rather than creating a new object.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`identifier`
</td>
<td>
the serialized form of the object.
</td>
</tr><tr>
<td>
`module_objects`
</td>
<td>
A dictionary of built-in objects to look the name up in.
Generally, `module_objects` is provided by midlevel library implementers.
</td>
</tr><tr>
<td>
`custom_objects`
</td>
<td>
A dictionary of custom objects to look the name up in.
Generally, `custom_objects` is provided by the end user.
</td>
</tr><tr>
<td>
`printable_module_name`
</td>
<td>
A human-readable string representing the type of the
object. Printed in case of exception.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The deserialized object.
</td>
</tr>

</table>



#### Example:



A mid-level library implementer might want to implement a utility for
retrieving an object from its config, as such:

```python
def deserialize(config, custom_objects=None):
   return deserialize_keras_object(
     identifier,
     module_objects=globals(),
     custom_objects=custom_objects,
     name="MyObjectType",
   )
```

This is how e.g. <a href="../../../tf/keras/layers/deserialize.md"><code>keras.layers.deserialize()</code></a> is implemented.