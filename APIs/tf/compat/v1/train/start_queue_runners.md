description: Starts all queue runners collected in the graph. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.train.start_queue_runners" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.train.start_queue_runners

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/training/queue_runner_impl.py">View source</a>



Starts all queue runners collected in the graph. (deprecated)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.train.queue_runner.start_queue_runners`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.train.start_queue_runners(
    sess=None, coord=None, daemon=(True), start=(True),
    collection=tf.GraphKeys.QUEUE_RUNNERS
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

QueueRunners are not compatible with eager execution. Instead, please
use [tf.data](https://www.tensorflow.org/guide/data) to get data into your
model.


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
To construct input pipelines, use the <a href="../../../../tf/data.md"><code>tf.data</code></a> module.

This is a companion method to `add_queue_runner()`.  It just starts
threads for all queue runners collected in the graph.  It returns
the list of all threads.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sess`
</td>
<td>
`Session` used to run the queue ops.  Defaults to the
default session.
</td>
</tr><tr>
<td>
`coord`
</td>
<td>
Optional `Coordinator` for coordinating the started threads.
</td>
</tr><tr>
<td>
`daemon`
</td>
<td>
Whether the threads should be marked as `daemons`, meaning
they don't block program exit.
</td>
</tr><tr>
<td>
`start`
</td>
<td>
Set to `False` to only create the threads, not start them.
</td>
</tr><tr>
<td>
`collection`
</td>
<td>
A `GraphKey` specifying the graph collection to
get the queue runners from.  Defaults to `GraphKeys.QUEUE_RUNNERS`.
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
if `sess` is None and there isn't any default session.
</td>
</tr><tr>
<td>
`TypeError`
</td>
<td>
if `sess` is not a <a href="../../../../tf/compat/v1/Session.md"><code>tf.compat.v1.Session</code></a> object.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of threads.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`RuntimeError`
</td>
<td>
If called with eager execution enabled.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If called without a default <a href="../../../../tf/compat/v1/Session.md"><code>tf.compat.v1.Session</code></a> registered.
</td>
</tr>
</table>

