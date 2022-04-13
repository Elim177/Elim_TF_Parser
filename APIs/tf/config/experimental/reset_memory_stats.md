description: Resets the tracked memory stats for the chosen device.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.config.experimental.reset_memory_stats" />
<meta itemprop="path" content="Stable" />
</div>

# tf.config.experimental.reset_memory_stats

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/config.py">View source</a>



Resets the tracked memory stats for the chosen device.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.config.experimental.reset_memory_stats`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.config.experimental.reset_memory_stats(
    device
)
</code></pre>



<!-- Placeholder for "Used in" -->

This function sets the tracked peak memory for a device to the device's
current memory usage. This allows you to measure the peak memory usage for a
specific part of your program. For example:

```
>>> if tf.config.list_physical_devices('GPU'):
...   # Sets the peak memory to the current memory.
...   tf.config.experimental.reset_memory_stats('GPU:0')
...   # Creates the first peak memory usage.
...   x1 = tf.ones(1000 * 1000, dtype=tf.float64)
...   del x1 # Frees the memory referenced by `x1`.
...   peak1 = tf.config.experimental.get_memory_info('GPU:0')['peak']
...   # Sets the peak memory to the current memory again.
...   tf.config.experimental.reset_memory_stats('GPU:0')
...   # Creates the second peak memory usage.
...   x2 = tf.ones(1000 * 1000, dtype=tf.float32)
...   del x2
...   peak2 = tf.config.experimental.get_memory_info('GPU:0')['peak']
...   assert peak2 < peak1  # tf.float32 consumes less memory than tf.float64.
```

Currently only supports GPU and TPU. If called on a CPU device, an exception
will be raised.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`device`
</td>
<td>
Device string to reset the memory stats, e.g. `"GPU:0"`, `"TPU:0"`.
See https://www.tensorflow.org/api_docs/python/tf/device for specifying
device strings.
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
No device found with the device name, like '"nonexistent"'.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
Invalid device name, like '"GPU"', '"CPU:GPU"', '"CPU:"'.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
Multiple devices matched with the device name.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
Memory statistics not tracked or clearing memory statistics not
supported, like '"CPU:0"'.
</td>
</tr>
</table>

