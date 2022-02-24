description: An ExtensionType that can be batched and unbatched.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.BatchableExtensionType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
</div>

# tf.experimental.BatchableExtensionType

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/extension_type.py">View source</a>



An ExtensionType that can be batched and unbatched.

Inherits From: [`ExtensionType`](../../tf/experimental/ExtensionType.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.experimental.BatchableExtensionType`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.BatchableExtensionType(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

`BatchableExtensionType`s can be used with APIs that require batching or
unbatching, including `Keras`, <a href="../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a>, and <a href="../../tf/map_fn.md"><code>tf.map_fn</code></a>.  E.g.:

```
>>> class Vehicle(BatchableExtensionType):
...   top_speed: tf.Tensor
...   mpg: tf.Tensor
>>> batch = Vehicle([120, 150, 80], [30, 40, 12])
>>> tf.map_fn(lambda vehicle: vehicle.top_speed * vehicle.mpg, batch,
...           fn_output_signature=tf.int32).numpy()
array([3600, 6000,  960], dtype=int32)
```

An `ExtensionTypeBatchEncoder` is used by these APIs to encode `ExtensionType`
values. The default encoder assumes that values can be stacked, unstacked, or
concatenated by simply stacking, unstacking, or concatenating every nested
`Tensor`, `ExtensionType`, `CompositeTensor`, or `TensorShape` field.
Extension types where this is not the case will need to override
`__batch_encoder__` with a custom `ExtensionTypeBatchEncoder`.  See
<a href="../../tf/experimental/ExtensionTypeBatchEncoder.md"><code>tf.experimental.ExtensionTypeBatchEncoder</code></a> for more details.

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/extension_type.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.


<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/extension_type.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Return self!=value.




