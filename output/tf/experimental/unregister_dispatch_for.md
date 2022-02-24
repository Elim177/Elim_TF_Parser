description: Unregisters a function that was registered with @dispatch_for_*.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.unregister_dispatch_for" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.unregister_dispatch_for

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/util/dispatch.py">View source</a>



Unregisters a function that was registered with `@dispatch_for_*`.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.experimental.unregister_dispatch_for`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.unregister_dispatch_for(
    dispatch_target
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is primarily intended for testing purposes.

#### Example:



```
>>> # Define a type and register a dispatcher to override `tf.abs`:
>>> class MyTensor(extension_type.ExtensionType):
...   value: tf.Tensor
>>> @dispatch_for_api(tf.abs)
... def my_abs(x: MyTensor):
...   return MyTensor(tf.abs(x.value))
>>> tf.abs(MyTensor(5))
MyTensor(value=<tf.Tensor: shape=(), dtype=int32, numpy=5>)
```

```
>>> # Unregister the dispatcher, so `tf.abs` no longer calls `my_abs`.
>>> unregister_dispatch_for(my_abs)
>>> tf.abs(MyTensor(5))
Traceback (most recent call last):
...
ValueError: Attempt to convert a value ... to a Tensor.
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dispatch_target`
</td>
<td>
The function to unregister.
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
If `dispatch_target` was not registered using `@dispatch_for`,
`@dispatch_for_unary_elementwise_apis`, or
`@dispatch_for_binary_elementwise_apis`.
</td>
</tr>
</table>

