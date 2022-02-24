description: Write a scalar summary.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.summary.scalar" />
<meta itemprop="path" content="Stable" />
</div>

# tf.summary.scalar

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tensorboard/tree/2.7.0/tensorboard/plugins/scalar/summary_v2.py#L30-L94">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Write a scalar summary.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.summary.scalar(
    name, data, step=None, description=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

See also <a href="../../tf/summary/image.md"><code>tf.summary.image</code></a>, <a href="../../tf/summary/histogram.md"><code>tf.summary.histogram</code></a>, <a href="../../tf/summary/SummaryWriter.md"><code>tf.summary.SummaryWriter</code></a>.

Writes simple numeric values for later analysis in TensorBoard.  Writes go to
the current default summary writer. Each summary point is associated with an
integral `step` value. This enables the incremental logging of time series
data.  A common usage of this API is to log loss during training to produce
a loss curve.

#### For example:



```python
test_summary_writer = tf.summary.create_file_writer('test/logdir')
with test_summary_writer.as_default():
    tf.summary.scalar('loss', 0.345, step=1)
    tf.summary.scalar('loss', 0.234, step=2)
    tf.summary.scalar('loss', 0.123, step=3)
```

Multiple independent time series may be logged by giving each series a unique
`name` value.

See [Get started with TensorBoard](https://www.tensorflow.org/tensorboard/get_started)
for more examples of effective usage of <a href="../../tf/summary/scalar.md"><code>tf.summary.scalar</code></a>.

In general, this API expects that data points are logged iwth a monotonically
increasing step value. Duplicate points for a single step or points logged out
of order by step are not guaranteed to display as desired in TensorBoard.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Arguments</h2></th></tr>

<tr>
<td>
`name`
</td>
<td>
A name for this summary. The summary tag used for TensorBoard will
be this name prefixed by any active name scopes.
</td>
</tr><tr>
<td>
`data`
</td>
<td>
A real numeric scalar value, convertible to a `float32` Tensor.
</td>
</tr><tr>
<td>
`step`
</td>
<td>
Explicit `int64`-castable monotonic step value for this summary. If
omitted, this defaults to <a href="../../tf/summary/experimental/get_step.md"><code>tf.summary.experimental.get_step()</code></a>, which must
not be None.
</td>
</tr><tr>
<td>
`description`
</td>
<td>
Optional long-form description for this summary, as a
constant `str`. Markdown is supported. Defaults to empty.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
True on success, or false if no summary was written because no default
summary writer was available.
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
if a default writer exists, but no step was provided and
<a href="../../tf/summary/experimental/get_step.md"><code>tf.summary.experimental.get_step()</code></a> is None.
</td>
</tr>
</table>

