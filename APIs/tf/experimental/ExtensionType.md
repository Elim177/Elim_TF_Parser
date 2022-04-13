description: Base class for TensorFlow ExtensionType classes.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.ExtensionType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
</div>

# tf.experimental.ExtensionType

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/extension_type.py">View source</a>



Base class for TensorFlow `ExtensionType` classes.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.experimental.ExtensionType`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.ExtensionType(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Tensorflow `ExtensionType` classes are specialized Python classes that can be
used transparently with TensorFlow -- e.g., they can be used with ops
such as <a href="../../tf/cond.md"><code>tf.cond</code></a> or <a href="../../tf/while_loop.md"><code>tf.while_loop</code></a> and used as inputs or outputs for
<a href="../../tf/function.md"><code>tf.function</code></a> and Keras layers.

New `ExtensionType` classes are defined by creating a subclass of
`tf.ExtensionType` that
contains type annotations for all instance variables.  The following type
annotations are supported:

Type                 | Example
-------------------- | --------------------------------------------
Python integers      | `i: int`
Python floats        | `f: float`
Python strings       | `s: str`
Python booleans      | `b: bool`
Python None          | `n: None`
Tensors              | `t: tf.Tensor`
Composite Tensors    | `rt: tf.RaggdTensor`
Extension Types      | `m: MyMaskedTensor`
Tensor shapes        | `shape: tf.TensorShape`
Tensor dtypes        | `dtype: tf.DType`
Type unions          | `length: typing.Union[int, float]`
Tuples               | `params: typing.Tuple[int, float, int, int]`
Tuples w/ Ellipsis   | `lengths: typing.Tuple[int, ...]`
Mappings             | `tags: typing.Mapping[str, str]`

Fields annotated with `typing.Mapping` will be stored using an immutable
mapping type.

ExtensionType values are immutable -- i.e., once constructed, you can not
modify or delete any of their instance members.

### Examples

```
>>> class MaskedTensor(ExtensionType):
...   values: tf.Tensor
...   mask: tf.Tensor
```

```
>>> class Toy(ExtensionType):
...   name: str
...   price: ops.Tensor
...   features: typing.Mapping[str, tf.Tensor]
```

```
>>> class ToyStore(ExtensionType):
...   name: str
...   toys: typing.Tuple[Toy, ...]
```

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/extension_type.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.


<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/extension_type.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Return self!=value.




