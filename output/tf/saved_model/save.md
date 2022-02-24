description: Exports a [tf.Module](https://www.tensorflow.org/api_docs/python/tf/Module) (and subclasses) obj to [SavedModel format](https://www.tensorflow.org/guide/saved_model#the_savedmodel_format_on_disk).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.saved_model.save" />
<meta itemprop="path" content="Stable" />
</div>

# tf.saved_model.save

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/saved_model/save.py">View source</a>



Exports a [tf.Module](https://www.tensorflow.org/api_docs/python/tf/Module) (and subclasses) `obj` to [SavedModel format](https://www.tensorflow.org/guide/saved_model#the_savedmodel_format_on_disk).

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.saved_model.experimental.save`, `tf.compat.v1.saved_model.save`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.saved_model.save(
    obj, export_dir, signatures=None, options=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The `obj` must inherit from the [`Trackable` class](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/tracking/base.py#L591).

#### Example usage:



```
>>> class Adder(tf.Module):
...   @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
...   def add(self, x):
...     return x + x
```

```
>>> model = Adder()
>>> tf.saved_model.save(model, '/tmp/adder')
```

The resulting SavedModel is then servable with an input named "x", a scalar
with dtype float32.

_Signatures_

Signatures define the input and output types for a computation. The optional
save `signatures` argument controls which methods in `obj` will be
available to programs which consume `SavedModel`s, for example, serving
APIs. Python functions may be decorated with
<a href="../../tf/function.md"><code>@tf.function(input_signature=...)</code></a> and passed as signatures directly, or
lazily with a call to `get_concrete_function` on the method decorated with
<a href="../../tf/function.md"><code>@tf.function</code></a>.

#### Example:



```
>>> class Adder(tf.Module):
...   @tf.function
...   def add(self, x):
...     return x + x
```

```
>>> model = Adder()
>>> tf.saved_model.save(
...   model, '/tmp/adder',signatures=model.add.get_concrete_function(
...     tf.TensorSpec([], tf.float32)))
```

If a <a href="../../tf/function.md"><code>@tf.function</code></a> does not have an input signature and
`get_concrete_function` is not called on that method, the function will not
be directly callable in the restored SavedModel.

#### Example:



```
>>> class Adder(tf.Module):
...   @tf.function
...   def add(self, x):
...     return x + x
```

```
>>> model = Adder()
>>> tf.saved_model.save(model, '/tmp/adder')
>>> restored = tf.saved_model.load('/tmp/adder')
>>> restored.add(1.)
Traceback (most recent call last):
...
ValueError: Found zero restored functions for caller function.
```

If the `signatures` argument is omitted, `obj` will be searched for
<a href="../../tf/function.md"><code>@tf.function</code></a>-decorated methods. If exactly one traced <a href="../../tf/function.md"><code>@tf.function</code></a> is
found, that method will be used as the default signature for the SavedModel.
Else, any <a href="../../tf/function.md"><code>@tf.function</code></a> attached to `obj` or its dependencies will be
exported for use with <a href="../../tf/saved_model/load.md"><code>tf.saved_model.load</code></a>.

When invoking a signature in an exported SavedModel, `Tensor` arguments are
identified by name. These names will come from the Python function's argument
names by default. They may be overridden by specifying a `name=...` argument
in the corresponding <a href="../../tf/TensorSpec.md"><code>tf.TensorSpec</code></a> object. Explicit naming is required if
multiple `Tensor`s are passed through a single argument to the Python
function.

The outputs of functions used as `signatures` must either be flat lists, in
which case outputs will be numbered, or a dictionary mapping string keys to
`Tensor`, in which case the keys will be used to name outputs.

Signatures are available in objects returned by <a href="../../tf/saved_model/load.md"><code>tf.saved_model.load</code></a> as a
`.signatures` attribute. This is a reserved attribute: <a href="../../tf/saved_model/save.md"><code>tf.saved_model.save</code></a>
on an object with a custom `.signatures` attribute will raise an exception.

_Using <a href="../../tf/saved_model/save.md"><code>tf.saved_model.save</code></a> with Keras models_

While Keras has its own [saving and loading API](https://www.tensorflow.org/guide/keras/save_and_serialize),
this function can be used to export Keras models. For example, exporting with
a signature specified:

```
>>> class Adder(tf.keras.Model):
...   @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
...   def concat(self, x):
...      return x + x
```

```
>>> model = Adder()
>>> tf.saved_model.save(model, '/tmp/adder')
```

Exporting from a function without a fixed signature:

```
>>> class Adder(tf.keras.Model):
...   @tf.function
...   def concat(self, x):
...      return x + x
```

```
>>> model = Adder()
>>> tf.saved_model.save(
...   model, '/tmp/adder',
...   signatures=model.concat.get_concrete_function(
...     tf.TensorSpec(shape=[], dtype=tf.string, name="string_input")))
```

<a href="../../tf/keras/Model.md"><code>tf.keras.Model</code></a> instances constructed from inputs and outputs already have a
signature and so do not require a <a href="../../tf/function.md"><code>@tf.function</code></a> decorator or a `signatures`
argument. If neither are specified, the model's forward pass is exported.

```
>>> x = tf.keras.layers.Input((4,), name="x")
>>> y = tf.keras.layers.Dense(5, name="out")(x)
>>> model = tf.keras.Model(x, y)
>>> tf.saved_model.save(model, '/tmp/saved_model/')
```

The exported SavedModel takes "x" with shape [None, 4] and returns "out"
with shape [None, 5]

_Variables and Checkpoints_

Variables must be tracked by assigning them to an attribute of a tracked
object or to an attribute of `obj` directly. TensorFlow objects (e.g. layers
from <a href="../../tf/keras/layers.md"><code>tf.keras.layers</code></a>, optimizers from <a href="../../tf/train.md"><code>tf.train</code></a>) track their variables
automatically. This is the same tracking scheme that <a href="../../tf/train/Checkpoint.md"><code>tf.train.Checkpoint</code></a>
uses, and an exported `Checkpoint` object may be restored as a training
checkpoint by pointing <a href="../../tf/train/Checkpoint.md#restore"><code>tf.train.Checkpoint.restore</code></a> to the SavedModel's
"variables/" subdirectory.

<a href="../../tf/function.md"><code>tf.function</code></a> does not hard-code device annotations from outside the function
body, instead of using the calling context's device. This means for example
that exporting a model that runs on a GPU and serving it on a CPU will
generally work, with some exceptions:

  * <a href="../../tf/device.md"><code>tf.device</code></a> annotations inside the body of the function will be hard-coded
    in the exported model; this type of annotation is discouraged.
  * Device-specific operations, e.g. with "cuDNN" in the name or with
    device-specific layouts, may cause issues.
  * For `ConcreteFunctions`, active distribution strategies will cause device
    placements to be hard-coded in the function.

SavedModels exported with <a href="../../tf/saved_model/save.md"><code>tf.saved_model.save</code></a> [strip default-valued
attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes)
automatically, which removes one source of incompatibilities when the consumer
of a SavedModel is running an older TensorFlow version than the
producer. There are however other sources of incompatibilities which are not
handled automatically, such as when the exported model contains operations
which the consumer does not have definitions for.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`obj`
</td>
<td>
A trackable object (e.g. tf.Module or tf.train.Checkpoint) to export.
</td>
</tr><tr>
<td>
`export_dir`
</td>
<td>
A directory in which to write the SavedModel.
</td>
</tr><tr>
<td>
`signatures`
</td>
<td>
Optional, one of three types:
* a <a href="../../tf/function.md"><code>tf.function</code></a> with an input signature specified, which will use the
  default serving signature key,
* the result of `f.get_concrete_function` on a <a href="../../tf/function.md"><code>@tf.function</code></a>-decorated
  function `f`, in which case `f` will be used to generate a signature for
  the SavedModel under the default serving signature key,
* a dictionary, which maps signature keys to either <a href="../../tf/function.md"><code>tf.function</code></a>
  instances with input signatures or concrete functions. Keys of such a
  dictionary may be arbitrary strings, but will typically be from the
  `tf.saved_model.signature_constants` module.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
<a href="../../tf/saved_model/SaveOptions.md"><code>tf.saved_model.SaveOptions</code></a> object for configuring save options.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If `obj` is not trackable.
</td>
</tr>
</table>




 <section><devsite-expandable expanded>
 <h2 class="showalways">eager compatibility</h2>

Not well supported when graph building. From TensorFlow 1.x,
<a href="../../tf/compat/v1/enable_eager_execution.md"><code>tf.compat.v1.enable_eager_execution()</code></a> should run first. Calling
tf.saved_model.save in a loop when graph building from TensorFlow 1.x will
add new save operations to the default graph each iteration.

May not be called from within a function body.


 </devsite-expandable></section>

