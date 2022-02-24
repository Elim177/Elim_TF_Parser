description: Base class for weight constraints.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.constraints.Constraint" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="get_config"/>
</div>

# tf.keras.constraints.Constraint

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/constraints.py#L27-L76">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Base class for weight constraints.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.constraints.Constraint`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

A `Constraint` instance works like a stateless function.
Users who subclass this
class should override the `__call__` method, which takes a single
weight parameter and return a projected version of that parameter
(e.g. normalized or clipped). Constraints can be used with various Keras
layers via the `kernel_constraint` or `bias_constraint` arguments.

Here's a simple example of a non-negative weight constraint:

```
>>> class NonNegative(tf.keras.constraints.Constraint):
...
...  def __call__(self, w):
...    return w * tf.cast(tf.math.greater_equal(w, 0.), w.dtype)
```

```
>>> weight = tf.constant((-1.0, 1.0))
>>> NonNegative()(weight)
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.,  1.], dtype=float32)>
```

```
>>> tf.keras.layers.Dense(4, kernel_constraint=NonNegative())
```

## Methods

<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/constraints.py#L67-L76">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

Returns a Python dict of the object config.

A constraint config is a Python dictionary (JSON-serializable) that can
be used to reinstantiate the same object.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Python dict containing the configuration of the constraint object.
</td>
</tr>

</table>



<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/constraints.py#L52-L65">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    w
)
</code></pre>

Applies the constraint to the input weight variable.

By default, the inputs weight variable is not modified.
Users should override this method to implement their own projection
function.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`w`
</td>
<td>
Input weight variable.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Projected variable (by default, returns unmodified inputs).
</td>
</tr>

</table>





