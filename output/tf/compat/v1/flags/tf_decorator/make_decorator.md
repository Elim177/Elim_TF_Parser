description: Make a decorator from a wrapper and a target.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.tf_decorator.make_decorator" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.flags.tf_decorator.make_decorator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/util/tf_decorator.py">View source</a>



Make a decorator from a wrapper and a target.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.app.flags.tf_decorator.make_decorator`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.tf_decorator.make_decorator(
    target, decorator_func, decorator_name=None, decorator_doc=&#x27;&#x27;,
    decorator_argspec=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`target`
</td>
<td>
The final callable to be wrapped.
</td>
</tr><tr>
<td>
`decorator_func`
</td>
<td>
The wrapper function.
</td>
</tr><tr>
<td>
`decorator_name`
</td>
<td>
The name of the decorator. If `None`, the name of the
function calling make_decorator.
</td>
</tr><tr>
<td>
`decorator_doc`
</td>
<td>
Documentation specific to this application of
`decorator_func` to `target`.
</td>
</tr><tr>
<td>
`decorator_argspec`
</td>
<td>
The new callable signature of this decorator.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The `decorator_func` argument with new metadata attached.
</td>
</tr>

</table>

