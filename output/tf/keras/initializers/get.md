description: Retrieve a Keras initializer by the identifier.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.initializers.get" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.initializers.get

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/initializers/__init__.py#L146-L193">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Retrieve a Keras initializer by the identifier.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.initializers.get`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.initializers.get`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.initializers.get(
    identifier
)
</code></pre>



<!-- Placeholder for "Used in" -->

The `identifier` may be the string name of a initializers function or class (
case-sensitively).

```
>>> identifier = 'Ones'
>>> tf.keras.initializers.deserialize(identifier)
<...keras.initializers.initializers_v2.Ones...>
```

You can also specify `config` of the initializer to this function by passing
dict containing `class_name` and `config` as an identifier. Also note that the
`class_name` must map to a `Initializer` class.

```
>>> cfg = {'class_name': 'Ones', 'config': {}}
>>> tf.keras.initializers.deserialize(cfg)
<...keras.initializers.initializers_v2.Ones...>
```

In the case that the `identifier` is a class, this method will return a new
instance of the class by its constructor.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`identifier`
</td>
<td>
String or dict that contains the initializer name or
configurations.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Initializer instance base on the input identifier.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If the input identifier is not a supported type or in a bad
format.
</td>
</tr>
</table>

