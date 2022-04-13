description: Convert a Keras model to dot format.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.model_to_dot" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.utils.model_to_dot

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/utils/vis_utils.py#L95-L350">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Convert a Keras model to dot format.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.utils.model_to_dot`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.model_to_dot(
    model, show_shapes=(False), show_dtype=(False), show_layer_names=(True),
    rankdir=&#x27;TB&#x27;, expand_nested=(False), dpi=96, subgraph=(False),
    layer_range=None, show_layer_activations=(False)
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model`
</td>
<td>
A Keras model instance.
</td>
</tr><tr>
<td>
`show_shapes`
</td>
<td>
whether to display shape information.
</td>
</tr><tr>
<td>
`show_dtype`
</td>
<td>
whether to display layer dtypes.
</td>
</tr><tr>
<td>
`show_layer_names`
</td>
<td>
whether to display layer names.
</td>
</tr><tr>
<td>
`rankdir`
</td>
<td>
`rankdir` argument passed to PyDot,
a string specifying the format of the plot:
'TB' creates a vertical plot;
'LR' creates a horizontal plot.
</td>
</tr><tr>
<td>
`expand_nested`
</td>
<td>
whether to expand nested models into clusters.
</td>
</tr><tr>
<td>
`dpi`
</td>
<td>
Dots per inch.
</td>
</tr><tr>
<td>
`subgraph`
</td>
<td>
whether to return a `pydot.Cluster` instance.
</td>
</tr><tr>
<td>
`layer_range`
</td>
<td>
input of `list` containing two `str` items, which is the
starting layer name and ending layer name (both inclusive) indicating
the range of layers for which the `pydot.Dot` will be generated. It
also accepts regex patterns instead of exact name. In such case, start
predicate will be the first element it matches to `layer_range[0]`
and the end predicate will be the last element it matches to
`layer_range[1]`. By default `None` which considers all layers of
model. Note that you must pass range such that the resultant subgraph
must be complete.
</td>
</tr><tr>
<td>
`show_layer_activations`
</td>
<td>
Display layer activations (only for layers that
have an `activation` property).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `pydot.Dot` instance representing the Keras model or
a `pydot.Cluster` instance representing nested model if
`subgraph=True`.
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
if `model_to_dot` is called before the model is built.
</td>
</tr><tr>
<td>
`ImportError`
</td>
<td>
if graphviz or pydot are not available.
</td>
</tr>
</table>

