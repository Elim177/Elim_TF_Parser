description: Get the current memory usage, in bytes, for the chosen device. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.config.experimental.get_memory_usage" />
<meta itemprop="path" content="Stable" />
</div>

# tf.config.experimental.get_memory_usage

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/config.py">View source</a>



Get the current memory usage, in bytes, for the chosen device. (deprecated)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.config.experimental.get_memory_usage`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.config.experimental.get_memory_usage(
    device
)
</code></pre>



<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.config.experimental.get_memory_info(device)['current'] instead.

This function is deprecated in favor of
<a href="../../../tf/config/experimental/get_memory_info.md"><code>tf.config.experimental.get_memory_info</code></a>. Calling this function is equivalent
to calling `tf.config.experimental.get_memory_info()['current']`.

See https://www.tensorflow.org/api_docs/python/tf/device for specifying device
strings.

#### For example:



```
>>> gpu_devices = tf.config.list_physical_devices('GPU')
>>> if gpu_devices:
...   tf.config.experimental.get_memory_usage('GPU:0')
```

Does not work for CPU.

For GPUs, TensorFlow will allocate all the memory by default, unless changed
with <a href="../../../tf/config/experimental/set_memory_growth.md"><code>tf.config.experimental.set_memory_growth</code></a>. This function only returns
the memory that TensorFlow is actually using, not the memory that TensorFlow
has allocated on the GPU.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`device`
</td>
<td>
Device string to get the bytes in use for, e.g. `"GPU:0"`
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Total memory usage in bytes.
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
Non-existent or CPU device specified.
</td>
</tr>
</table>

