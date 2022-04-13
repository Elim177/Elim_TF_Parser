description: Clear pending operations and error statuses in async execution.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.async_clear_error" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.async_clear_error

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/eager/context.py">View source</a>



Clear pending operations and error statuses in async execution.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.experimental.async_clear_error`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.async_clear_error()
</code></pre>



<!-- Placeholder for "Used in" -->

In async execution mode, an error in op/function execution can lead to errors
in subsequent ops/functions that are scheduled but not yet executed. Calling
this method clears all pending operations and reset the async execution state.

#### Example:



```
while True:
  try:
    # Step function updates the metric `loss` internally
    train_step_fn()
  except tf.errors.OutOfRangeError:
    tf.experimental.async_clear_error()
    break
logging.info('loss = %s', loss.numpy())
```