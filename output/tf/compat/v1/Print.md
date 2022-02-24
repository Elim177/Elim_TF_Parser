description: Prints a list of tensors. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.Print" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.Print

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/logging_ops.py">View source</a>



Prints a list of tensors. (deprecated)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.Print(
    input_, data, message=None, first_n=None, summarize=None, name=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

This API is deprecated. Use <a href="../../../tf/print.md"><code>tf.print</code></a> instead. <a href="../../../tf/print.md"><code>tf.print</code></a> does not need the
`input_` argument.

<a href="../../../tf/print.md"><code>tf.print</code></a> works in TF2 when executing eagerly and inside a <a href="../../../tf/function.md"><code>tf.function</code></a>.

In TF1-styled sessions, an explicit control dependency declaration is needed
to execute the <a href="../../../tf/print.md"><code>tf.print</code></a> operation. Refer to the documentation of
<a href="../../../tf/print.md"><code>tf.print</code></a> for more details.


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2018-08-20.
Instructions for updating:
Use tf.print instead of tf.Print. Note that tf.print returns a no-output operator that directly prints the output. Outside of defuns or eager mode, this operator will not be executed unless it is directly specified in session.run or used as a control dependency for other operators. This is only a concern in graph mode. Below is an example of how to ensure tf.print executes in graph mode:


This is an identity op (behaves like <a href="../../../tf/identity.md"><code>tf.identity</code></a>) with the side effect
of printing `data` when evaluating.

Note: This op prints to the standard error. It is not currently compatible
  with jupyter notebook (printing to the notebook *server's* output, not into
  the notebook).



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_`
</td>
<td>
A tensor passed through this op.
</td>
</tr><tr>
<td>
`data`
</td>
<td>
A list of tensors to print out when op is evaluated.
</td>
</tr><tr>
<td>
`message`
</td>
<td>
A string, prefix of the error message.
</td>
</tr><tr>
<td>
`first_n`
</td>
<td>
Only log `first_n` number of times. Negative numbers log always;
this is the default.
</td>
</tr><tr>
<td>
`summarize`
</td>
<td>
Only print this many entries of each tensor. If None, then a
maximum of 3 elements are printed per input tensor.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor`. Has the same type and contents as `input_`.

```python
sess = tf.compat.v1.Session()
with sess.as_default():
    tensor = tf.range(10)
    print_op = tf.print(tensor)
    with tf.control_dependencies([print_op]):
      out = tf.add(tensor, tensor)
    sess.run(out)
```
</td>
</tr>

</table>

