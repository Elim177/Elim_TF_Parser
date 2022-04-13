description: Scope class for bfloat16 variables so that the model uses custom getter.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.tpu.bfloat16_scope" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.tpu.bfloat16_scope

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/tpu/bfloat16.py">View source</a>



Scope class for bfloat16 variables so that the model uses custom getter.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@tf_contextlib.contextmanager</code>
<code>tf.compat.v1.tpu.bfloat16_scope(
    name: Optional[Text] = None
) -> Generator[<a href="../../../../tf/compat/v1/variable_scope.md"><code>tf.compat.v1.variable_scope</code></a>, None, None]
</code></pre>



<!-- Placeholder for "Used in" -->

This enables variables to be read as bfloat16 type when using get_variable.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Arguments</h2></th></tr>

<tr>
<td>
`name`
</td>
<td>
Name to use for scope.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Yields</h2></th></tr>
<tr class="alt">
<td colspan="2">
a variable scope.
</td>
</tr>

</table>

