description: Turns logging for device placement decisions on or off.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.debugging.set_log_device_placement" />
<meta itemprop="path" content="Stable" />
</div>

# tf.debugging.set_log_device_placement

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/eager/context.py">View source</a>



Turns logging for device placement decisions on or off.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.debugging.set_log_device_placement`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.debugging.set_log_device_placement(
    enabled
)
</code></pre>



<!-- Placeholder for "Used in" -->

Operations execute on a particular device, producing and consuming tensors on
that device. This may change the performance of the operation or require
TensorFlow to copy data to or from an accelerator, so knowing where operations
execute is useful for debugging performance issues.

For more advanced profiling, use the [TensorFlow
profiler](https://www.tensorflow.org/guide/profiler).

Device placement for operations is typically controlled by a <a href="../../tf/device.md"><code>tf.device</code></a>
scope, but there are exceptions, for example operations on a <a href="../../tf/Variable.md"><code>tf.Variable</code></a>
which follow the initial placement of the variable. Turning off soft device
placement (with <a href="../../tf/config/set_soft_device_placement.md"><code>tf.config.set_soft_device_placement</code></a>) provides more explicit
control.

```
>>> tf.debugging.set_log_device_placement(True)
>>> tf.ones([])
>>> # [...] op Fill in device /job:localhost/replica:0/task:0/device:GPU:0
>>> with tf.device("CPU"):
...  tf.ones([])
>>> # [...] op Fill in device /job:localhost/replica:0/task:0/device:CPU:0
>>> tf.debugging.set_log_device_placement(False)
```

Turning on <a href="../../tf/debugging/set_log_device_placement.md"><code>tf.debugging.set_log_device_placement</code></a> also logs the placement of
ops inside <a href="../../tf/function.md"><code>tf.function</code></a> when the function is called.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`enabled`
</td>
<td>
Whether to enabled device placement logging.
</td>
</tr>
</table>

