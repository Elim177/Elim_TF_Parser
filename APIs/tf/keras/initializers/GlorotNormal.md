description: The Glorot normal initializer, also called Xavier normal initializer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.initializers.GlorotNormal" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tf.keras.initializers.GlorotNormal

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/initializers/initializers_v2.py#L728-L771">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The Glorot normal initializer, also called Xavier normal initializer.

Inherits From: [`VarianceScaling`](../../../tf/keras/initializers/VarianceScaling.md), [`Initializer`](../../../tf/keras/initializers/Initializer.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.initializers.GlorotNormal`, `tf.initializers.glorot_normal`, `tf.keras.initializers.glorot_normal`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.initializers.GlorotNormal(
    seed=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Also available via the shortcut function
<a href="../../../tf/keras/initializers/GlorotNormal.md"><code>tf.keras.initializers.glorot_normal</code></a>.

Draws samples from a truncated normal distribution centered on 0 with `stddev
= sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of input units in
the weight tensor and `fan_out` is the number of output units in the weight
tensor.

#### Examples:



```
>>> # Standalone usage:
>>> initializer = tf.keras.initializers.GlorotNormal()
>>> values = initializer(shape=(2, 2))
```

```
>>> # Usage in a Keras layer:
>>> initializer = tf.keras.initializers.GlorotNormal()
>>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`seed`
</td>
<td>
A Python integer. Used to create random seeds. See
<a href="../../../tf/compat/v1/set_random_seed.md"><code>tf.compat.v1.set_random_seed</code></a> for behavior. Note that seeded
initializer will not produce same random values across multiple calls,
but multiple initializers will produce same sequence when constructed with
same seed value.
</td>
</tr>
</table>



#### References:

- [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)


## Methods

<h3 id="from_config"><code>from_config</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/initializers/initializers_v2.py#L88-L107">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_config(
    config
)
</code></pre>

Instantiates an initializer from a configuration dictionary.


#### Example:



```python
initializer = RandomUniform(-1, 1)
config = initializer.get_config()
initializer = RandomUniform.from_config(config)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`config`
</td>
<td>
A Python dictionary, the output of `get_config`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../../tf/keras/initializers/Initializer.md"><code>tf.keras.initializers.Initializer</code></a> instance.
</td>
</tr>

</table>



<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/initializers/initializers_v2.py#L770-L771">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

Returns the configuration of the initializer as a JSON-serializable dict.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A JSON-serializable Python dict.
</td>
</tr>

</table>



<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/initializers/initializers_v2.py#L502-L534">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    shape, dtype=None, **kwargs
)
</code></pre>

Returns a tensor object initialized as specified by the initializer.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`shape`
</td>
<td>
Shape of the tensor.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
Optional dtype of the tensor. Only floating point types are
supported. If not specified, <a href="../../../tf/keras/backend/floatx.md"><code>tf.keras.backend.floatx()</code></a> is used, which
default to `float32` unless you configured it otherwise (via
<a href="../../../tf/keras/backend/set_floatx.md"><code>tf.keras.backend.set_floatx(float_dtype)</code></a>)
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Additional keyword arguments.
</td>
</tr>
</table>




