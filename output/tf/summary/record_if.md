description: Sets summary recording on or off per the provided boolean value.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.summary.record_if" />
<meta itemprop="path" content="Stable" />
</div>

# tf.summary.record_if

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/summary_ops_v2.py">View source</a>



Sets summary recording on or off per the provided boolean value.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@tf_contextlib.contextmanager</code>
<code>tf.summary.record_if(
    condition
)
</code></pre>



<!-- Placeholder for "Used in" -->

The provided value can be a python boolean, a scalar boolean Tensor, or
or a callable providing such a value; if a callable is passed it will be
invoked on-demand to determine whether summary writing will occur.  Note that
when calling record_if() in an eager mode context, if you intend to provide a
varying condition like `step % 100 == 0`, you must wrap this in a
callable to avoid immediate eager evaluation of the condition.  In particular,
using a callable is the only way to have your condition evaluated as part of
the traced body of an @tf.function that is invoked from within the
`record_if()` context.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`condition`
</td>
<td>
can be True, False, a bool Tensor, or a callable providing such.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Yields</h2></th></tr>
<tr class="alt">
<td colspan="2">
Returns a context manager that sets this value on enter and restores the
previous value on exit.
</td>
</tr>

</table>

