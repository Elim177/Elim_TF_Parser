description: Asserts that the given condition is true.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.debugging.Assert" />
<meta itemprop="path" content="Stable" />
</div>

# tf.debugging.Assert

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/control_flow_ops.py">View source</a>



Asserts that the given condition is true.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.Assert`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.Assert`, `tf.compat.v1.debugging.Assert`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.debugging.Assert(
    condition, data, summarize=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

If `condition` evaluates to false, print the list of tensors in `data`.
`summarize` determines how many entries of the tensors to print.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`condition`
</td>
<td>
The condition to evaluate.
</td>
</tr><tr>
<td>
`data`
</td>
<td>
The tensors to print out when condition is false.
</td>
</tr><tr>
<td>
`summarize`
</td>
<td>
Print this many entries of each tensor.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for this operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`assert_op`
</td>
<td>
An `Operation` that, when executed, raises a
<a href="../../tf/errors/InvalidArgumentError.md"><code>tf.errors.InvalidArgumentError</code></a> if `condition` is not true.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>


</table>


Note: The output of this function should be used. If it is not, a warning will be logged or an error may be raised. To mark the output as used, call its .mark_used() method.

 <section><devsite-expandable expanded>
 <h2 class="showalways">TF1 compatibility</h2>

  When in TF V1 mode (that is, outside <a href="../../tf/function.md"><code>tf.function</code></a>) Assert needs a control
  dependency on the output to ensure the assertion executes:

```python
# Ensure maximum element of x is smaller or equal to 1
assert_op = tf.Assert(tf.less_equal(tf.reduce_max(x), 1.), [x])
with tf.control_dependencies([assert_op]):
  ... code using x ...
```



 </devsite-expandable></section>



 <section><devsite-expandable expanded>
 <h2 class="showalways">eager compatibility</h2>

returns None


 </devsite-expandable></section>

