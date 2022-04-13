description: Maintains moving averages of variables by employing an exponential decay.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.train.ExponentialMovingAverage" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="apply"/>
<meta itemprop="property" content="average"/>
</div>

# tf.train.ExponentialMovingAverage

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/training/moving_averages.py">View source</a>



Maintains moving averages of variables by employing an exponential decay.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.train.ExponentialMovingAverage`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.train.ExponentialMovingAverage(
    decay, num_updates=None, zero_debias=(False),
    name=&#x27;ExponentialMovingAverage&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

When training a model, it is often beneficial to maintain moving averages of
the trained parameters.  Evaluations that use averaged parameters sometimes
produce significantly better results than the final trained values.

The `apply()` method adds shadow copies of trained variables the first time
it is called, and maintains a moving average of the trained variables in
their shadow copies at every additional invocation.
It should generally be called immediately after creating the model weights,
and then after each training step.

The `average()` method gives access to the shadow variables.
It allows you to use the moving averages in place of the last trained values
for evaluations, by loading the moving averages into your model via
`var.assign(ema.average(var))`.
Additionally, although `ExponentialMovingAverage`
objects are not directly trackable by checkpoints,
`average()` returns the moving average variables for your model weights,
which you can then checkpoint. (There is an example
of this near the bottom of this docstring).
So, `average()` is useful when
building an evaluation model, or when restoring a model from a checkpoint
file.

The moving averages are computed using exponential decay.  You specify the
decay value when creating the `ExponentialMovingAverage` object.  The shadow
variables are initialized with the same initial values as the trained
variables.  When you run `apply` to maintain the moving averages, each
shadow variable is updated with the formula:

  `shadow_variable -= (1 - decay) * (shadow_variable - variable)`

This is mathematically equivalent to the classic formula below, but the use
of an `assign_sub` op (the `"-="` in the formula) allows concurrent lockless
updates to the variables:

  `shadow_variable = decay * shadow_variable + (1 - decay) * variable`

Reasonable values for `decay` are close to 1.0, typically in the
multiple-nines range: 0.999, 0.9999, etc.

Example usage when creating a training model:

```python
# Create variables.
var0 = tf.Variable(...)
var1 = tf.Variable(...)
# ... use the variables to build a training model...

# Create an ExponentialMovingAverage object
ema = tf.train.ExponentialMovingAverage(decay=0.9999)

# The first `apply` creates the shadow variables that hold the moving averages
ema.apply([var0, var1])

# grab the moving averages for checkpointing purposes or to be able to
# load the moving averages into the model weights
averages = [ema.average(var0), ema.average(var1)]

...
def train_step(...):
...
  # Apply the optimizer.
  opt.minimize(my_loss, [var0, var1])

  # Update the moving averages
  # of var0 and var1 with additional calls to `apply`
  ema.apply([var0, var1])

...train the model by running train_step multiple times...
```

There are several ways to use the moving averages for evaluations:

1. Assign the values of the shadow variables to your model variables with
   <a href="../../tf/Variable.md#assign"><code>Variable.assign(...)</code></a> before evaluating your
   model. You can use the `average()`
   method to get the shadow variable for a given variable. To continue
   training after using this approach, make sure to record the unaveraged
   weights and restore them before continuing to train. You can see the
   tensorflow-addons' MovingAverage optimizer's `swap_weights` method for
   one example of how to swap variables efficiently in distributed settings:
   https://github.com/tensorflow/addons/blob/v0.13.0/tensorflow_addons/optimizers/moving_average.py#L151
2. Make sure to checkpoint out your moving average variables in your
   <a href="../../tf/train/Checkpoint.md"><code>tf.train.Checkpoint</code></a>. At evaluation time, create your shadow variables and
   use <a href="../../tf/train/Checkpoint.md"><code>tf.train.Checkpoint</code></a> to restore the moving averages into the shadow
   variables. Then, load the moving averages into the actual model weights via
   `var.assign(moving_avg)`.
3. Checkpoint out your moving average variables in your <a href="../../tf/train/Checkpoint.md"><code>tf.train.Checkpoint</code></a>.
   For evaluation, restore your model weights directly from the moving
   averages instead of from the non-averaged weights.
   Caution: If you choose this approach, include only the object-graph paths
   to the averaged path in your checkpoint restore.
   If you point both the unaveraged and averaged paths in a checkpoint
   restore to the same variables, it is hard to reason about whether your
   model will restore the averaged or non-averaged variables.

Example of saving out then restoring the shadow variable values:

```python
# Create variables.
var0 = tf.Variable(...)
var1 = tf.Variable(...)
# ... use the variables to build a training model...

# Create an ExponentialMovingAverage object, create the shadow variables,
# and grab the moving averages for checkpointing purposes.
# (The ExponentialMovingAverage object itself is not checkpointable)
ema = tf.train.ExponentialMovingAverage(decay=0.9999)
ema.apply([var0, var1])
avg_var0 = ema.average(var0)
avg_var1 = ema.average(var1)

# Create a Checkpoint that will manage the model weights and the averages,
checkpoint = tf.train.Checkpoint(model_weights=[var0, var1],
                                 averaged_weights=[avg_var0, avg_var1])
... # Do training

# Save out the checkpoint including the model weights and the moving averages
checkpoint.save(...)
```

Restore option: restore all averaged & non-averaged weights, then load
moving averages into the model via `var.assign()`
```python
# Create variables.
var0 = tf.Variable(...)
var1 = tf.Variable(...)
# ... use the variables to build a training model...

# Create an ExponentialMovingAverage object, create the shadow variables,
# and grab the moving averages for checkpoint restore purposes.
# (The ExponentialMovingAverage object itself is not checkpointable)
ema = tf.train.ExponentialMovingAverage(decay=0.9999)
ema.apply([var0, var1])
avg_var0 = ema.average(var0)
avg_var1 = ema.average(var1)

# Create a Checkpoint that will manage the model weights and the averages,
checkpoint = tf.train.Checkpoint(model_weights=[var0, var1],
                                 averaged_weights=[avg_var0, avg_var1])
checkpoint.restore(...)
var0.assign(avg_var0)
var1.assign(avg_var1)
# var0 and var1 now hold the moving average values
```

Restore option: Directly restore the moving averages into the model weights.
```python
# Create variables.
var0 = tf.Variable(...)
var1 = tf.Variable(...)
# ... use the variables to build a training model...

# Create a Checkpoint that will manage two objects with trackable state,
checkpoint = tf.train.Checkpoint(averaged_weights=[var0, var1])
checkpoint.restore(...)
# var0 and var1 now hold the moving average values
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`decay`
</td>
<td>
Float.  The decay to use.
</td>
</tr><tr>
<td>
`num_updates`
</td>
<td>
Optional count of number of updates applied to variables.
</td>
</tr><tr>
<td>
`zero_debias`
</td>
<td>
If `True`, zero debias moving-averages that are initialized
with tensors. (Note: moving averages may not be initialized with
non-variable tensors when eager execution is enabled).
</td>
</tr><tr>
<td>
`name`
</td>
<td>
String. Optional prefix name to use for the name of ops added in
`apply()`.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`name`
</td>
<td>
The name of this ExponentialMovingAverage object.
</td>
</tr>
</table>



## Methods

<h3 id="apply"><code>apply</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/training/moving_averages.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>apply(
    var_list=None
)
</code></pre>

Maintains moving averages of variables.

`var_list` must be a list of `Variable` objects.  This method
creates shadow variables (holding the moving averages)
for all elements of `var_list`, and
updates the moving averages using the current `var_list` values. Shadow
variables for `Variable` objects are initialized to the variable's initial
value.

Shadow variables are created with `trainable=False`. To access them you
can use the EMA object's `average` method. Note that `EMA` objects are
not trackable by checkpoints, so if you want to checkpoint or restore the
moving variables you will need to manually grab the shadow
variables via `average()` and assign them as <a href="../../tf/Module.md"><code>tf.Module</code></a> properties or
directly pass them to your <a href="../../tf/train/Checkpoint.md"><code>tf.train.Checkpoint</code></a>.

Note that `apply()` can be called multiple times. When eager execution is
enabled each call to apply will update the variables once, so this needs to
be called in a loop.

In legacy TF 1.x graphs, this method returns an op that updates all
shadow variables from the current value of their associated variables. In
TF 1.x graphs without automatically control dependencies this op needs to be
manually run.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`var_list`
</td>
<td>
A list of Variable objects. The variables
must be of types bfloat16, float16, float32, or float64.
(In legacy TF 1.x graphs these may be tensors, but this is unsupported
when eager execution is enabled.)
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An Operation that updates the moving averages.
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
If the arguments are not an allowed type.
</td>
</tr>
</table>



<h3 id="average"><code>average</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/training/moving_averages.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>average(
    var
)
</code></pre>

Returns the `Variable` holding the average of `var`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`var`
</td>
<td>
A `Variable` object.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Variable` object or `None` if the moving average of `var`
is not maintained.
</td>
</tr>

</table>





