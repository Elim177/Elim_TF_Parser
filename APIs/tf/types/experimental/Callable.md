description: Base class for TF callables like those created by tf.function.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.types.experimental.Callable" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
</div>

# tf.types.experimental.Callable

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/types/core.py">View source</a>



Base class for TF callables like those created by tf.function.

<!-- Placeholder for "Used in" -->

Note: Callables are conceptually very similar to <a href="../../../tf/Operation.md"><code>tf.Operation</code></a>: a
<a href="../../../tf/Operation.md"><code>tf.Operation</code></a> is a kind of callable.

## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/types/core.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    *args, **kwargs
)
</code></pre>

Executes this callable.

This behaves like a regular op - in eager mode, it immediately starts
execution, returning results. In graph mode, it creates ops which return
symbolic TensorFlow values (like <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>, <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a>,
etc.). For example, <a href="../../../tf/function.md"><code>tf.function</code></a> callables typically generate a
<a href="../../../tf/raw_ops/PartitionedCall.md"><code>tf.raw_ops.PartitionedCall</code></a> op, but not always - the
exact operations being generated are an internal implementation detail.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*args`
</td>
<td>
positional argument for this call
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
keyword arguments for this call
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The execution results.
</td>
</tr>

</table>





