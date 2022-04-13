description: Options for constructing a Checkpoint.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.train.CheckpointOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="experimental_io_device"/>
</div>

# tf.train.CheckpointOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/training/saving/checkpoint_options.py">View source</a>



Options for constructing a Checkpoint.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.train.CheckpointOptions`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.train.CheckpointOptions(
    experimental_io_device=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Used as the `options` argument to either <a href="../../tf/train/Checkpoint.md#save"><code>tf.train.Checkpoint.save()</code></a> or
<a href="../../tf/train/Checkpoint.md#restore"><code>tf.train.Checkpoint.restore()</code></a> methods to adjust how variables are
saved/restored.

Example: Run IO ops on "localhost" while saving a checkpoint:

```
step = tf.Variable(0, name="step")
checkpoint = tf.train.Checkpoint(step=step)
options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
checkpoint.save("/tmp/ckpt", options=options)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`experimental_io_device`
</td>
<td>
string. Applies in a distributed setting.
Tensorflow device to use to access the filesystem. If `None` (default)
then for each variable the filesystem is accessed from the CPU:0 device
of the host where that variable is assigned. If specified, the
filesystem is instead accessed from that device for all variables.

This is for example useful if you want to save to a local directory,
such as "/tmp" when running in a distributed setting. In that case pass
a device for the host where the "/tmp" directory is accessible.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
experimental_io_device<a id="experimental_io_device"></a>
</td>
<td>
Instance of `member_descriptor`
</td>
</tr>
</table>

