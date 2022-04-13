description: An error is raised for unsupported operator in Graph execution.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.errors.OperatorNotAllowedInGraphError" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf.errors.OperatorNotAllowedInGraphError

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/errors_impl.py">View source</a>



An error is raised for unsupported operator in Graph execution.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.errors.OperatorNotAllowedInGraphError(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

For example, using a <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a> as a Python `bool` in Graph execution
is not allowed.

