description: Serialize the optimizer configuration to JSON compatible python dict.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.optimizers.serialize" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.optimizers.serialize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/optimizers.py#L40-L58">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Serialize the optimizer configuration to JSON compatible python dict.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.optimizers.serialize`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.optimizers.serialize`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.optimizers.serialize(
    optimizer
)
</code></pre>



<!-- Placeholder for "Used in" -->

The configuration can be used for persistence and reconstruct the `Optimizer`
instance again.

```
>>> tf.keras.optimizers.serialize(tf.keras.optimizers.SGD())
{'class_name': 'SGD', 'config': {'name': 'SGD', 'learning_rate': 0.01,
                                 'decay': 0.0, 'momentum': 0.0,
                                 'nesterov': False}}
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`optimizer`
</td>
<td>
An `Optimizer` instance to serialize.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Python dict which contains the configuration of the input optimizer.
</td>
</tr>

</table>

