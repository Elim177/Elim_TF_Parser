description: Optimizer that implements the Adadelta algorithm.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.train.AdadeltaOptimizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="apply_gradients"/>
<meta itemprop="property" content="compute_gradients"/>
<meta itemprop="property" content="get_name"/>
<meta itemprop="property" content="get_slot"/>
<meta itemprop="property" content="get_slot_names"/>
<meta itemprop="property" content="minimize"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="GATE_GRAPH"/>
<meta itemprop="property" content="GATE_NONE"/>
<meta itemprop="property" content="GATE_OP"/>
</div>

# tf.compat.v1.train.AdadeltaOptimizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/training/adadelta.py">View source</a>



Optimizer that implements the Adadelta algorithm.

Inherits From: [`Optimizer`](../../../../tf/compat/v1/train/Optimizer.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.train.AdadeltaOptimizer(
    learning_rate=0.001, rho=0.95, epsilon=1e-08, use_locking=(False),
    name=&#x27;Adadelta&#x27;
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

tf.compat.v1.train.AdadeltaOptimizer is compatible with eager mode and
<a href="../../../../tf/function.md"><code>tf.function</code></a>.
When eager execution is enabled, `learning_rate`, `rho`,
and `epsilon` can each be a callable that
takes no arguments and returns the actual value to use. This can be useful
for changing these values across different invocations of optimizer
functions.

To switch to native TF2 style, use [`tf.keras.optimizers.Adadelta`]
(https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adadelta)
instead. Please notice that due to the implementation differences,
<a href="../../../../tf/keras/optimizers/Adadelta.md"><code>tf.keras.optimizers.Adadelta</code></a> and
<a href="../../../../tf/compat/v1/train/AdadeltaOptimizer.md"><code>tf.compat.v1.train.AdadeltaOptimizer</code></a> may have slight differences in
floating point numerics even though the formula used for the variable
updates still matches.

#### Structural mapping to native TF2

Before:

```python
optimizer = tf.compat.v1.train.AdadeltaOptimizer(
  learning_rate=learning_rate,
  rho=rho,
  epsilon=epsilon)
```

After:

```python
optimizer = tf.keras.optimizers.Adadelta(
  learning_rate=learning_rate,
  rho=rho,
  epsilon=epsilon)
```

#### How to map arguments
| TF1 Arg Name       | TF2 Arg Name   | Note                             |
| ------------------ | -------------  | -------------------------------  |
| `learning_rate`    | `learning_rate`| Be careful of setting           |
: : : learning_rate tensor value computed from the global step.          :
: : : In TF1 this was usually meant to imply a dynamic learning rate and :
: : : would recompute in each step. In TF2 (eager + function) it will    :
: : : treat it as a scalar value that only gets computed once instead of :
: : : a symbolic placeholder to be computed each time.                   :
| `rho`              | `rho`          | -                                |
| `epsilon`          | `epsilon`      | Default value is 1e-08 in TF1,   |
:                    :                : but 1e-07 in TF2.                :
| `use_locking`      | -              | Not applicable in TF2.           |

#### Before & after usage example
Before:

```python
x = tf.Variable([1,2,3], dtype=tf.float32)
grad = tf.constant([0.1, 0.2, 0.3])
optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=0.001)
optimizer.apply_gradients(zip([grad], [x]))
```

After:

```python
x = tf.Variable([1,2,3], dtype=tf.float32)
grad = tf.constant([0.1, 0.2, 0.3])
optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.001)
optimizer.apply_gradients(zip([grad], [x]))
```



 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->


#### References:

ADADELTA - An Adaptive Learning Rate Method:
  [Zeiler, 2012](http://arxiv.org/abs/1212.5701)
  ([pdf](http://arxiv.org/pdf/1212.5701v1.pdf))




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`learning_rate`
</td>
<td>
A `Tensor` or a floating point value. The learning rate.
To match the exact form in the original paper use 1.0.
</td>
</tr><tr>
<td>
`rho`
</td>
<td>
A `Tensor` or a floating point value. The decay rate.
</td>
</tr><tr>
<td>
`epsilon`
</td>
<td>
A `Tensor` or a floating point value.  A constant epsilon used
to better conditioning the grad update.
</td>
</tr><tr>
<td>
`use_locking`
</td>
<td>
If `True` use locks for update operations.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optional name prefix for the operations created when applying
gradients.  Defaults to "Adadelta".
</td>
</tr>
</table>



## Methods

<h3 id="apply_gradients"><code>apply_gradients</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/training/optimizer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>apply_gradients(
    grads_and_vars, global_step=None, name=None
)
</code></pre>

Apply gradients to variables.

This is the second part of `minimize()`. It returns an `Operation` that
applies gradients.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`grads_and_vars`
</td>
<td>
List of (gradient, variable) pairs as returned by
`compute_gradients()`.
</td>
</tr><tr>
<td>
`global_step`
</td>
<td>
Optional `Variable` to increment by one after the
variables have been updated.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optional name for the returned operation.  Default to the
name passed to the `Optimizer` constructor.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An `Operation` that applies the specified gradients. If `global_step`
was not None, that operation also increments `global_step`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
If `grads_and_vars` is malformed.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If none of the variables have gradients.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
If you should use `_distributed_apply()` instead.
</td>
</tr>
</table>



<h3 id="compute_gradients"><code>compute_gradients</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/training/optimizer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute_gradients(
    loss, var_list=None, gate_gradients=GATE_OP, aggregation_method=None,
    colocate_gradients_with_ops=(False), grad_loss=None
)
</code></pre>

Compute gradients of `loss` for the variables in `var_list`.

This is the first part of `minimize()`.  It returns a list
of (gradient, variable) pairs where "gradient" is the gradient
for "variable".  Note that "gradient" can be a `Tensor`, an
`IndexedSlices`, or `None` if there is no gradient for the
given variable.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`loss`
</td>
<td>
A Tensor containing the value to minimize or a callable taking
no arguments which returns the value to minimize. When eager execution
is enabled it must be a callable.
</td>
</tr><tr>
<td>
`var_list`
</td>
<td>
Optional list or tuple of <a href="../../../../tf/Variable.md"><code>tf.Variable</code></a> to update to minimize
`loss`.  Defaults to the list of variables collected in the graph
under the key `GraphKeys.TRAINABLE_VARIABLES`.
</td>
</tr><tr>
<td>
`gate_gradients`
</td>
<td>
How to gate the computation of gradients.  Can be
`GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
</td>
</tr><tr>
<td>
`aggregation_method`
</td>
<td>
Specifies the method used to combine gradient terms.
Valid values are defined in the class `AggregationMethod`.
</td>
</tr><tr>
<td>
`colocate_gradients_with_ops`
</td>
<td>
If True, try colocating gradients with
the corresponding op.
</td>
</tr><tr>
<td>
`grad_loss`
</td>
<td>
Optional. A `Tensor` holding the gradient computed for `loss`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of (gradient, variable) pairs. Variable is always present, but
gradient can be `None`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
If `var_list` contains anything else than `Variable` objects.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If some arguments are invalid.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
If called with eager execution enabled and `loss` is
not callable.
</td>
</tr>
</table>




 <section><devsite-expandable expanded>
 <h4 class="showalways">eager compatibility</h4>

When eager execution is enabled, `gate_gradients`, `aggregation_method`,
and `colocate_gradients_with_ops` are ignored.


 </devsite-expandable></section>



<h3 id="get_name"><code>get_name</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/training/optimizer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_name()
</code></pre>




<h3 id="get_slot"><code>get_slot</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/training/optimizer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_slot(
    var, name
)
</code></pre>

Return a slot named `name` created for `var` by the Optimizer.

Some `Optimizer` subclasses use additional variables.  For example
`Momentum` and `Adagrad` use variables to accumulate updates.  This method
gives access to these `Variable` objects if for some reason you need them.

Use `get_slot_names()` to get the list of slot names created by the
`Optimizer`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`var`
</td>
<td>
A variable passed to `minimize()` or `apply_gradients()`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A string.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The `Variable` for the slot if it was created, `None` otherwise.
</td>
</tr>

</table>



<h3 id="get_slot_names"><code>get_slot_names</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/training/optimizer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_slot_names()
</code></pre>

Return a list of the names of slots created by the `Optimizer`.

See `get_slot()`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of strings.
</td>
</tr>

</table>



<h3 id="minimize"><code>minimize</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/training/optimizer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>minimize(
    loss, global_step=None, var_list=None, gate_gradients=GATE_OP,
    aggregation_method=None, colocate_gradients_with_ops=(False), name=None,
    grad_loss=None
)
</code></pre>

Add operations to minimize `loss` by updating `var_list`.

This method simply combines calls `compute_gradients()` and
`apply_gradients()`. If you want to process the gradient before applying
them call `compute_gradients()` and `apply_gradients()` explicitly instead
of using this function.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`loss`
</td>
<td>
A `Tensor` containing the value to minimize.
</td>
</tr><tr>
<td>
`global_step`
</td>
<td>
Optional `Variable` to increment by one after the
variables have been updated.
</td>
</tr><tr>
<td>
`var_list`
</td>
<td>
Optional list or tuple of `Variable` objects to update to
minimize `loss`.  Defaults to the list of variables collected in
the graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
</td>
</tr><tr>
<td>
`gate_gradients`
</td>
<td>
How to gate the computation of gradients.  Can be
`GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
</td>
</tr><tr>
<td>
`aggregation_method`
</td>
<td>
Specifies the method used to combine gradient terms.
Valid values are defined in the class `AggregationMethod`.
</td>
</tr><tr>
<td>
`colocate_gradients_with_ops`
</td>
<td>
If True, try colocating gradients with
the corresponding op.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optional name for the returned operation.
</td>
</tr><tr>
<td>
`grad_loss`
</td>
<td>
Optional. A `Tensor` holding the gradient computed for `loss`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An Operation that updates the variables in `var_list`.  If `global_step`
was not `None`, that operation also increments `global_step`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If some of the variables are not `Variable` objects.
</td>
</tr>
</table>




 <section><devsite-expandable expanded>
 <h4 class="showalways">eager compatibility</h4>

When eager execution is enabled, `loss` should be a Python function that
takes no arguments and computes the value to be minimized. Minimization (and
gradient computation) is done with respect to the elements of `var_list` if
not None, else with respect to any trainable variables created during the
execution of the `loss` function. `gate_gradients`, `aggregation_method`,
`colocate_gradients_with_ops` and `grad_loss` are ignored when eager
execution is enabled.


 </devsite-expandable></section>



<h3 id="variables"><code>variables</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/training/optimizer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>variables()
</code></pre>

A list of variables which encode the current state of `Optimizer`.

Includes slot variables and additional global variables created by the
optimizer in the current default graph.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of variables.
</td>
</tr>

</table>







<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
GATE_GRAPH<a id="GATE_GRAPH"></a>
</td>
<td>
`2`
</td>
</tr><tr>
<td>
GATE_NONE<a id="GATE_NONE"></a>
</td>
<td>
`0`
</td>
</tr><tr>
<td>
GATE_OP<a id="GATE_OP"></a>
</td>
<td>
`1`
</td>
</tr>
</table>

