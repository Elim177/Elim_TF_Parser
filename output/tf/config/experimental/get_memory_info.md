description: Get memory info for the chosen device, as a dict.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.config.experimental.get_memory_info" />
<meta itemprop="path" content="Stable" />
</div>

# tf.config.experimental.get_memory_info

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/config.py">View source</a>



Get memory info for the chosen device, as a dict.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.config.experimental.get_memory_info`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.config.experimental.get_memory_info(
    device
)
</code></pre>



<!-- Placeholder for "Used in" -->

This function returns a dict containing information about the device's memory
usage. For example:

```
>>> if tf.config.list_physical_devices('GPU'):
...   # Returns a dict in the form {'current': <current mem usage>,
...   #                             'peak': <peak mem usage>}
...   tf.config.experimental.get_memory_info('GPU:0')
```

Currently returns the following keys:
  - `'current'`: The current memory used by the device, in bytes.
  - `'peak'`: The peak memory used by the device across the run of the
      program, in bytes. Can be reset with
      <a href="../../../tf/config/experimental/reset_memory_stats.md"><code>tf.config.experimental.reset_memory_stats</code></a>.

More keys may be added in the future, including device-specific keys.

Currently only supports GPU and TPU. If called on a CPU device, an exception
will be raised.

For GPUs, TensorFlow will allocate all the memory by default, unless changed
with <a href="../../../tf/config/experimental/set_memory_growth.md"><code>tf.config.experimental.set_memory_growth</code></a>. The dict specifies only the
current and peak memory that TensorFlow is actually using, not the memory that
TensorFlow has allocated on the GPU.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`device`
</td>
<td>
Device string to get the memory information for, e.g. `"GPU:0"`,
`"TPU:0"`. See https://www.tensorflow.org/api_docs/python/tf/device for
  specifying device strings.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dict with keys `'current'` and `'peak'`, specifying the current and peak
memory usage respectively.
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
Memory statistics not tracked, like '"CPU:0"'.
</td>
</tr>
</table>

