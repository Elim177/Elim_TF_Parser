description: Stops gradient computation.
robots: noindex

# tf.raw_ops.StopGradient

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Stops gradient computation.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.StopGradient`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.StopGradient(
    input, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

When executed in a graph, this op outputs its input tensor as-is.

When building ops to compute gradients, this op prevents the contribution of
its inputs to be taken into account.  Normally, the gradient generator adds ops
to a graph to compute the derivatives of a specified 'loss' by recursively
finding out inputs that contributed to its computation.  If you insert this op
in the graph it inputs are masked from the gradient generator.  They are not
taken into account for computing gradients.

This is useful any time you want to compute a value with TensorFlow but need
to pretend that the value was a constant. For example, the softmax function
for a vector x can be written as

```python

  def softmax(x):
    numerator = tf.exp(x)
    denominator = tf.reduce_sum(numerator)
    return numerator / denominator
```

This however is susceptible to overflow if the values in x are large. An
alternative more stable way is to subtract the maximum of x from each of the
values.

```python

  def stable_softmax(x):
    z = x - tf.reduce_max(x)
    numerator = tf.exp(z)
    denominator = tf.reduce_sum(numerator)
    return numerator / denominator
```

However, when we backprop through the softmax to x, we dont want to backprop
through the <a href="../../tf/math/reduce_max.md"><code>tf.reduce_max(x)</code></a> (if the max values are not unique then the
gradient could flow to the wrong input) calculation and treat that as a
constant. Therefore, we should write this out as

```python

  def stable_softmax(x):
    z = x - tf.stop_gradient(tf.reduce_max(x))
    numerator = tf.exp(z)
    denominator = tf.reduce_sum(numerator)
    return numerator / denominator
```

Some other examples include:

*  The *EM* algorithm where the *M-step* should not involve backpropagation
   through the output of the *E-step*.
*  Contrastive divergence training of Boltzmann machines where, when
   differentiating the energy function, the training must not backpropagate
   through the graph that generated the samples from the model.
*  Adversarial training, where no backprop should happen through the adversarial
   example generation process.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`
</td>
<td>
A `Tensor`.
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
A `Tensor`. Has the same type as `input`.
</td>
</tr>

</table>

