description: Serialize a Keras object into a JSON-compatible representation.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.serialize_keras_object" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.utils.serialize_keras_object

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/utils/generic_utils.py#L467-L533">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Serialize a Keras object into a JSON-compatible representation.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.utils.serialize_keras_object`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.serialize_keras_object(
    instance
)
</code></pre>



<!-- Placeholder for "Used in" -->

Calls to `serialize_keras_object` while underneath the
`SharedObjectSavingScope` context manager will cause any objects re-used
across multiple layers to be saved with a special shared object ID. This
allows the network to be re-created properly during deserialization.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`instance`
</td>
<td>
The object to serialize.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dict-like, JSON-compatible representation of the object's config.
</td>
</tr>

</table>

