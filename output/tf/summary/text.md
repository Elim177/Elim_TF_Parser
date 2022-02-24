description: Write a text summary.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.summary.text" />
<meta itemprop="path" content="Stable" />
</div>

# tf.summary.text

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tensorboard/tree/2.7.0/tensorboard/plugins/text/summary_v2.py#L26-L97">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Write a text summary.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.summary.text(
    name, data, step=None, description=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

See also <a href="../../tf/summary/scalar.md"><code>tf.summary.scalar</code></a>, <a href="../../tf/summary/SummaryWriter.md"><code>tf.summary.SummaryWriter</code></a>, <a href="../../tf/summary/image.md"><code>tf.summary.image</code></a>.

Writes text Tensor values for later visualization and analysis in TensorBoard.
Writes go to the current default summary writer.  Like <a href="../../tf/summary/scalar.md"><code>tf.summary.scalar</code></a>
points, text points are each associated with a `step` and a `name`.
All the points with the same `name` constitute a time series of text values.

#### For Example:


```python
test_summary_writer = tf.summary.create_file_writer('test/logdir')
with test_summary_writer.as_default():
    tf.summary.text('first_text', 'hello world!', step=0)
    tf.summary.text('first_text', 'nice to meet you!', step=1)
```

The text summary can also contain Markdown, and TensorBoard will render the text
as such.

```python
with test_summary_writer.as_default():
    text_data = '''
          | *hello* | *there* |
          |---------|---------|
          | this    | is      |
          | a       | table   |
    '''
    text_data = '\n'.join(l.strip() for l in text_data.splitlines())
    tf.summary.text('markdown_text', text_data, step=0)
```

Since text is Tensor valued, each text point may be a Tensor of string values.
rank-1 and rank-2 Tensors are rendered as tables in TensorBoard.  For higher ranked
Tensors, you'll see just a 2D slice of the data.  To avoid this, reshape the Tensor
to at most rank-2 prior to passing it to this function.

Demo notebook at
["Displaying text data in TensorBoard"](https://www.tensorflow.org/tensorboard/text_summaries).

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
A UTF-8 string Tensor value.
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
True on success, or false if no summary was emitted because no default
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

