description: Clone a Functional or Sequential Model instance.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.models.clone_model" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.models.clone_model

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/models.py#L388-L457">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Clone a Functional or Sequential `Model` instance.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.models.clone_model`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.models.clone_model(
    model, input_tensors=None, clone_function=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Model cloning is similar to calling a model on new inputs,
except that it creates new layers (and thus new weights) instead
of sharing the weights of the existing layers.

Note that
`clone_model` will not preserve the uniqueness of shared objects within the
model (e.g. a single variable attached to two distinct layers will be
restored as two separate variables).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model`
</td>
<td>
Instance of `Model`
(could be a Functional model or a Sequential model).
</td>
</tr><tr>
<td>
`input_tensors`
</td>
<td>
optional list of input tensors or InputLayer objects
to build the model upon. If not provided,
new `Input` objects will be created.
</td>
</tr><tr>
<td>
`clone_function`
</td>
<td>
Callable to be used to clone each layer in the target
model (except `InputLayer` instances). It takes as argument the layer
instance to be cloned, and returns the corresponding layer instance to
be used in the model copy. If unspecified, this callable defaults to
the following serialization/deserialization function:
`lambda layer: layer.__class__.from_config(layer.get_config())`.
By passing a custom callable, you can customize your copy of the
model, e.g. by wrapping certain layers of interest (you might want to
replace all `LSTM` instances with equivalent
`Bidirectional(LSTM(...))` instances, for example).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An instance of `Model` reproducing the behavior
of the original model, on top of new inputs tensors,
using newly instantiated weights. The cloned model may behave
differently from the original model if a custom `clone_function`
modifies the layer.
</td>
</tr>

</table>



#### Example:



```python
# Create a test Sequential model.
model = keras.Sequential([
    keras.Input(shape=(728,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])
# Create a copy of the test model (with freshly initialized weights).
new_model = clone_model(model)
```

Note that subclassed models cannot be cloned, since their internal
layer structure is not known. To achieve equivalent functionality
as `clone_model` in the case of a subclassed model, simply make sure
that the model class implements `get_config()`
(and optionally `from_config()`), and call:

```python
new_model = model.__class__.from_config(model.get_config())
```