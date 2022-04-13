description: Compiles a function into a callable TensorFlow graph. (deprecated arguments)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.function" />
<meta itemprop="path" content="Stable" />
</div>

# tf.function

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/eager/def_function.py">View source</a>



Compiles a function into a callable TensorFlow graph. (deprecated arguments)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.function`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.function(
    func=None, input_signature=None, autograph=(True), jit_compile=None,
    experimental_implements=None, experimental_autograph_options=None,
    experimental_relax_shapes=(False), experimental_compile=None,
    experimental_follow_type_hints=None
) -> <a href="../tf/types/experimental/GenericFunction.md"><code>tf.types.experimental.GenericFunction</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

Warning: SOME ARGUMENTS ARE DEPRECATED: `(experimental_compile)`. They will be removed in a future version.
Instructions for updating:
experimental_compile is deprecated, use jit_compile instead

<a href="../tf/function.md"><code>tf.function</code></a> constructs a <a href="../tf/types/experimental/GenericFunction.md"><code>tf.types.experimental.GenericFunction</code></a> that
executes a TensorFlow graph (<a href="../tf/Graph.md"><code>tf.Graph</code></a>) created by trace-compiling the
TensorFlow operations in `func`. More information on the topic can be found
in [Introduction to Graphs and tf.function]
(https://www.tensorflow.org/guide/intro_to_graphs).

See [Better Performance with tf.function]
(https://www.tensorflow.org/guide/function) for tips on performance and
known limitations.

#### Example usage:



```
>>> @tf.function
... def f(x, y):
...   return x ** 2 + y
>>> x = tf.constant([2, 3])
>>> y = tf.constant([3, -2])
>>> f(x, y)
<tf.Tensor: ... numpy=array([7, 7], ...)>
```

The trace-compilation allows non-TensorFlow operations to execute, but under
special conditions. In general, only TensorFlow operations are guaranteed to
run and create fresh results whenever the `GenericFunction` is called.

## Features

`func` may use data-dependent control flow, including `if`, `for`, `while`
`break`, `continue` and `return` statements:

```
>>> @tf.function
... def f(x):
...   if tf.reduce_sum(x) > 0:
...     return x * x
...   else:
...     return -x // 2
>>> f(tf.constant(-2))
<tf.Tensor: ... numpy=1>
```

`func`'s closure may include <a href="../tf/Tensor.md"><code>tf.Tensor</code></a> and <a href="../tf/Variable.md"><code>tf.Variable</code></a> objects:

```
>>> @tf.function
... def f():
...   return x ** 2 + y
>>> x = tf.constant([-2, -3])
>>> y = tf.Variable([3, -2])
>>> f()
<tf.Tensor: ... numpy=array([7, 7], ...)>
```

`func` may also use ops with side effects, such as <a href="../tf/print.md"><code>tf.print</code></a>, <a href="../tf/Variable.md"><code>tf.Variable</code></a>
and others:

```
>>> v = tf.Variable(1)
>>> @tf.function
... def f(x):
...   for i in tf.range(x):
...     v.assign_add(i)
>>> f(3)
>>> v
<tf.Variable ... numpy=4>
```

Important: Any Python side-effects (appending to a list, printing with
`print`, etc) will only happen once, when `func` is traced. To have
side-effects executed into your <a href="../tf/function.md"><code>tf.function</code></a> they need to be written
as TF ops:

```
>>> l = []
>>> @tf.function
... def f(x):
...   for i in x:
...     l.append(i + 1)    # Caution! Will only happen once when tracing
>>> f(tf.constant([1, 2, 3]))
>>> l
[<tf.Tensor ...>]
```

Instead, use TensorFlow collections like <a href="../tf/TensorArray.md"><code>tf.TensorArray</code></a>:

```
>>> @tf.function
... def f(x):
...   ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
...   for i in range(len(x)):
...     ta = ta.write(i, x[i] + 1)
...   return ta.stack()
>>> f(tf.constant([1, 2, 3]))
<tf.Tensor: ..., numpy=array([2, 3, 4], ...)>
```

## <a href="../tf/function.md"><code>tf.function</code></a> creates polymorphic callables

Internally, <a href="../tf/types/experimental/GenericFunction.md"><code>tf.types.experimental.GenericFunction</code></a> may contain multiple
<a href="../tf/types/experimental/ConcreteFunction.md"><code>tf.types.experimental.ConcreteFunction</code></a>s, each specialized to arguments with
different data types or shapes, since TensorFlow can perform more
optimizations on graphs of specific shapes, dtypes and values of constant
arguments. <a href="../tf/function.md"><code>tf.function</code></a> treats any pure Python values as opaque objects (best
thought of as compile-time constants), and builds a separate <a href="../tf/Graph.md"><code>tf.Graph</code></a> for
each set of Python arguments that it encounters.
For more information, see the
[tf.function guide](https://www.tensorflow.org/guide/function#rules_of_tracing)

Executing a `GenericFunction` will select and execute the appropriate
`ConcreteFunction` based on the argument types and values.

To obtain an individual `ConcreteFunction`, use the
<a href="../tf/types/experimental/GenericFunction.md#get_concrete_function"><code>GenericFunction.get_concrete_function</code></a> method. It can be called with the
same arguments as `func` and returns a
<a href="../tf/types/experimental/ConcreteFunction.md"><code>tf.types.experimental.ConcreteFunction</code></a>. `ConcreteFunction`s are backed by a
single <a href="../tf/Graph.md"><code>tf.Graph</code></a>:

```
>>> @tf.function
... def f(x):
...   return x + 1
>>> isinstance(f.get_concrete_function(1).graph, tf.Graph)
True
```

`ConcreteFunction`s can be executed just like `GenericFunction`s, but their
input is resticted to the types to which they're specialized.

## Retracing

`ConcreteFunctions` are built (traced) on the fly, as the `GenericFunction` is
called with new TensorFlow types or shapes, or with new Python values as
arguments. When `GenericFunction` builds a new trace, it is said that `func`
is retraced. Retracing is a frequent performance concern for <a href="../tf/function.md"><code>tf.function</code></a> as
it can be considerably slower than executing a graph that's already been
traced. It is ideal to minimize the amount of retracing in your code.

Caution: Passing python scalars or lists as arguments to <a href="../tf/function.md"><code>tf.function</code></a> will
usually retrace. To avoid this, pass numeric arguments as Tensors whenever
possible:

```
>>> @tf.function
... def f(x):
...   return tf.abs(x)
>>> f1 = f.get_concrete_function(1)
>>> f2 = f.get_concrete_function(2)  # Slow - compiles new graph
>>> f1 is f2
False
>>> f1 = f.get_concrete_function(tf.constant(1))
>>> f2 = f.get_concrete_function(tf.constant(2))  # Fast - reuses f1
>>> f1 is f2
True
```

Python numerical arguments should only be used when they take few distinct
values, such as hyperparameters like the number of layers in a neural network.

## Input signatures

For Tensor arguments, `GenericFunction`creates a new `ConcreteFunction` for
every unique set of input shapes and datatypes. The example below creates two
separate `ConcreteFunction`s, each specialized to a different shape:

```
>>> @tf.function
... def f(x):
...   return x + 1
>>> vector = tf.constant([1.0, 1.0])
>>> matrix = tf.constant([[3.0]])
>>> f.get_concrete_function(vector) is f.get_concrete_function(matrix)
False
```

An "input signature" can be optionally provided to <a href="../tf/function.md"><code>tf.function</code></a> to control
this process. The input signature specifies the shape and type of each
Tensor argument to the function using a <a href="../tf/TensorSpec.md"><code>tf.TensorSpec</code></a> object. More general
shapes can be used. This ensures only one `ConcreteFunction` is created, and
restricts the `GenericFunction` to the specified shapes and types. It is
an effective way to limit retracing when Tensors have dynamic shapes.

```
>>> @tf.function(
...     input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
... def f(x):
...   return x + 1
>>> vector = tf.constant([1.0, 1.0])
>>> matrix = tf.constant([[3.0]])
>>> f.get_concrete_function(vector) is f.get_concrete_function(matrix)
True
```

## Variables may only be created once

<a href="../tf/function.md"><code>tf.function</code></a> only allows creating new <a href="../tf/Variable.md"><code>tf.Variable</code></a> objects when it is called
for the first time:

```
>>> class MyModule(tf.Module):
...   def __init__(self):
...     self.v = None
...
...   @tf.function
...   def __call__(self, x):
...     if self.v is None:
...       self.v = tf.Variable(tf.ones_like(x))
...     return self.v * x
```

In general, it is recommended to create <a href="../tf/Variable.md"><code>tf.Variable</code></a>s outside of
<a href="../tf/function.md"><code>tf.function</code></a>.
In simple cases, persisting state across <a href="../tf/function.md"><code>tf.function</code></a> boundaries may be
implemented using a pure functional style in which state is represented by
<a href="../tf/Tensor.md"><code>tf.Tensor</code></a>s passed as arguments and returned as return values.

Contrast the two styles below:

```
>>> state = tf.Variable(1)
>>> @tf.function
... def f(x):
...   state.assign_add(x)
>>> f(tf.constant(2))  # Non-pure functional style
>>> state
<tf.Variable ... numpy=3>
```

```
>>> state = tf.constant(1)
>>> @tf.function
... def f(state, x):
...   state += x
...   return state
>>> state = f(state, tf.constant(2))  # Pure functional style
>>> state
<tf.Tensor: ... numpy=3>
```

## Python operations execute only once per trace

`func` may contain TensorFlow operations mixed with pure Python operations.
However, when the function is executed, only the TensorFlow operations will
run. The Python operations run only once, at trace time. If TensorFlow
operations depend on results from Pyhton operations, those results will be
frozen into the graph.

```
>>> @tf.function
... def f(a, b):
...   print('this runs at trace time; a is', a, 'and b is', b)
...   return b
>>> f(1, tf.constant(1))
this runs at trace time; a is 1 and b is Tensor("...", shape=(), dtype=int32)
<tf.Tensor: shape=(), dtype=int32, numpy=1>
```

```
>>> f(1, tf.constant(2))
<tf.Tensor: shape=(), dtype=int32, numpy=2>
```

```
>>> f(2, tf.constant(1))
this runs at trace time; a is 2 and b is Tensor("...", shape=(), dtype=int32)
<tf.Tensor: shape=(), dtype=int32, numpy=1>
```

```
>>> f(2, tf.constant(2))
<tf.Tensor: shape=(), dtype=int32, numpy=2>
```

## Using type annotations to improve performance

'experimental_follow_type_hints` can be used along with type annotations to
reduce retracing by automatically casting any Python values to <a href="../tf/Tensor.md"><code>tf.Tensor</code></a>
(something that is not done by default, unless you use input signatures).

```
>>> @tf.function(experimental_follow_type_hints=True)
... def f_with_hints(x: tf.Tensor):
...   print('Tracing')
...   return x
>>> @tf.function(experimental_follow_type_hints=False)
... def f_no_hints(x: tf.Tensor):
...   print('Tracing')
...   return x
>>> f_no_hints(1)
Tracing
<tf.Tensor: shape=(), dtype=int32, numpy=1>
>>> f_no_hints(2)
Tracing
<tf.Tensor: shape=(), dtype=int32, numpy=2>
>>> f_with_hints(1)
Tracing
<tf.Tensor: shape=(), dtype=int32, numpy=1>
>>> f_with_hints(2)
<tf.Tensor: shape=(), dtype=int32, numpy=2>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`func`
</td>
<td>
the function to be compiled. If `func` is None, <a href="../tf/function.md"><code>tf.function</code></a> returns
a decorator that can be invoked with a single argument - `func`. In other
words, `tf.function(input_signature=...)(func)` is equivalent to
<a href="../tf/function.md"><code>tf.function(func, input_signature=...)</code></a>. The former can be used as
decorator.
</td>
</tr><tr>
<td>
`input_signature`
</td>
<td>
A possibly nested sequence of <a href="../tf/TensorSpec.md"><code>tf.TensorSpec</code></a> objects
specifying the shapes and dtypes of the Tensors that will be supplied to
this function. If `None`, a separate function is instantiated for each
inferred input signature.  If input_signature is specified, every input to
`func` must be a `Tensor`, and `func` cannot accept `**kwargs`.
</td>
</tr><tr>
<td>
`autograph`
</td>
<td>
Whether autograph should be applied on `func` before tracing a
graph. Data-dependent control flow requires `autograph=True`. For more
information, see the [tf.function and AutoGraph guide](
https://www.tensorflow.org/guide/function#autograph_transformations).
</td>
</tr><tr>
<td>
`jit_compile`
</td>
<td>
If `True`, compiles the function using
[XLA](https://tensorflow.org/xla). XLA performs compiler optimizations,
such as fusion, and attempts to emit more efficient code. This may
drastically improve the performance. If set to `True`,
the whole function needs to be compilable by XLA, or an
<a href="../tf/errors/InvalidArgumentError.md"><code>errors.InvalidArgumentError</code></a> is thrown.
If `None` (default), compiles the function with XLA when running on TPU
and goes through the regular function execution path when running on
other devices.
If `False`, executes the function without XLA compilation.  Set this value
to `False` when directly running a multi-device function on TPUs (e.g. two
TPU cores, one TPU core and its host CPU).
Not all functions are compilable, see a list of
[sharp corners](https://tensorflow.org/xla/known_issues).
</td>
</tr><tr>
<td>
`experimental_implements`
</td>
<td>
If provided, contains a name of a "known" function
this implements. For example "mycompany.my_recurrent_cell".
This is stored as an attribute in inference function,
which can then be detected when processing serialized function.
See [standardizing composite ops](https://github.com/tensorflow/community/blob/master/rfcs/20190610-standardizing-composite_ops.md)  
for details.  For an example of utilizing this attribute see this
[example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc)
The code above automatically detects and substitutes function that
implements "embedded_matmul" and allows TFLite to substitute its own
implementations. For instance, a tensorflow user can use this
 attribute to mark that their function also implements
`embedded_matmul` (perhaps more efficiently!)
by specifying it using this parameter:
`@tf.function(experimental_implements="embedded_matmul")`
This can either be specified as just the string name of the function or
a NameAttrList corresponding to a list of key-value attributes associated
with the function name. The name of the function will be in the 'name'
field of the NameAttrList. To define a formal TF op for this function
implements, try the experimental [composite TF](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/tfr)
project.
</td>
</tr><tr>
<td>
`experimental_autograph_options`
</td>
<td>
Optional tuple of
<a href="../tf/autograph/experimental/Feature.md"><code>tf.autograph.experimental.Feature</code></a> values.
</td>
</tr><tr>
<td>
`experimental_relax_shapes`
</td>
<td>
When True, <a href="../tf/function.md"><code>tf.function</code></a> may generate fewer,
graphs that are less specialized on input shapes.
</td>
</tr><tr>
<td>
`experimental_compile`
</td>
<td>
Deprecated alias to 'jit_compile'.
</td>
</tr><tr>
<td>
`experimental_follow_type_hints`
</td>
<td>
When True, the function may use type
annotations from `func` to optimize the tracing performance. For example,
arguments annotated with <a href="../tf/Tensor.md"><code>tf.Tensor</code></a> will automatically be converted
to a Tensor.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
If `func` is not None, returns a <a href="../tf/types/experimental/GenericFunction.md"><code>tf.types.experimental.GenericFunction</code></a>.
If `func` is None, returns a decorator that, when invoked with a single
`func` argument, returns a <a href="../tf/types/experimental/GenericFunction.md"><code>tf.types.experimental.GenericFunction</code></a>.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>
<tr class="alt">
<td colspan="2">
`ValueError` when attempting to use `jit_compile=True`, but XLA support is
not available.
</td>
</tr>

</table>

