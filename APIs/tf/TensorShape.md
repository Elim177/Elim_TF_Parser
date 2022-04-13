description: Represents the shape of a Tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.TensorShape" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__bool__"/>
<meta itemprop="property" content="__concat__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__nonzero__"/>
<meta itemprop="property" content="__radd__"/>
<meta itemprop="property" content="as_list"/>
<meta itemprop="property" content="as_proto"/>
<meta itemprop="property" content="assert_has_rank"/>
<meta itemprop="property" content="assert_is_compatible_with"/>
<meta itemprop="property" content="assert_is_fully_defined"/>
<meta itemprop="property" content="assert_same_rank"/>
<meta itemprop="property" content="concatenate"/>
<meta itemprop="property" content="is_compatible_with"/>
<meta itemprop="property" content="is_fully_defined"/>
<meta itemprop="property" content="merge_with"/>
<meta itemprop="property" content="most_specific_compatible_shape"/>
<meta itemprop="property" content="num_elements"/>
<meta itemprop="property" content="with_rank"/>
<meta itemprop="property" content="with_rank_at_least"/>
<meta itemprop="property" content="with_rank_at_most"/>
</div>

# tf.TensorShape

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>



Represents the shape of a `Tensor`.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.TensorShape`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.TensorShape(
    dims
)
</code></pre>



<!-- Placeholder for "Used in" -->

A `TensorShape` represents a possibly-partial shape specification for a
`Tensor`. It may be one of the following:

* *Fully-known shape:* has a known number of dimensions and a known size
  for each dimension. e.g. `TensorShape([16, 256])`
* *Partially-known shape:* has a known number of dimensions, and an unknown
  size for one or more dimension. e.g. `TensorShape([None, 256])`
* *Unknown shape:* has an unknown number of dimensions, and an unknown
  size in all dimensions. e.g. `TensorShape(None)`

If a tensor is produced by an operation of type `"Foo"`, its shape
may be inferred if there is a registered shape function for
`"Foo"`. See [Shape
functions](https://tensorflow.org/extend/adding_an_op#shape_functions_in_c)
for details of shape functions and how to register them. Alternatively,
the shape may be set explicitly using <a href="../tf/Tensor.md#set_shape"><code>tf.Tensor.set_shape</code></a>.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dims`
</td>
<td>
A list of Dimensions, or None if the shape is unspecified.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
If dims cannot be converted to a list of dimensions.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`dims`
</td>
<td>
Deprecated.  Returns list of dimensions for this shape.

Suggest <a href="../tf/TensorShape.md#as_list"><code>TensorShape.as_list</code></a> instead.
</td>
</tr><tr>
<td>
`ndims`
</td>
<td>
Deprecated accessor for `rank`.
</td>
</tr><tr>
<td>
`rank`
</td>
<td>
Returns the rank of this shape, or None if it is unspecified.
</td>
</tr>
</table>



## Methods

<h3 id="as_list"><code>as_list</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_list()
</code></pre>

Returns a list of integers or `None` for each dimension.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of integers or `None` for each dimension.
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
If `self` is an unknown shape with an unknown rank.
</td>
</tr>
</table>



<h3 id="as_proto"><code>as_proto</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_proto()
</code></pre>

Returns this shape as a `TensorShapeProto`.


<h3 id="assert_has_rank"><code>assert_has_rank</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>assert_has_rank(
    rank
)
</code></pre>

Raises an exception if `self` is not compatible with the given `rank`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`rank`
</td>
<td>
An integer.
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
If `self` does not represent a shape with the given `rank`.
</td>
</tr>
</table>



<h3 id="assert_is_compatible_with"><code>assert_is_compatible_with</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>assert_is_compatible_with(
    other
)
</code></pre>

Raises exception if `self` and `other` do not represent the same shape.

This method can be used to assert that there exists a shape that both
`self` and `other` represent.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another TensorShape.
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
If `self` and `other` do not represent the same shape.
</td>
</tr>
</table>



<h3 id="assert_is_fully_defined"><code>assert_is_fully_defined</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>assert_is_fully_defined()
</code></pre>

Raises an exception if `self` is not fully defined in every dimension.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If `self` does not have a known value for every dimension.
</td>
</tr>
</table>



<h3 id="assert_same_rank"><code>assert_same_rank</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>assert_same_rank(
    other
)
</code></pre>

Raises an exception if `self` and `other` do not have compatible ranks.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another `TensorShape`.
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
If `self` and `other` do not represent shapes with the
same rank.
</td>
</tr>
</table>



<h3 id="concatenate"><code>concatenate</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>concatenate(
    other
)
</code></pre>

Returns the concatenation of the dimension in `self` and `other`.

*N.B.* If either `self` or `other` is completely unknown,
concatenation will discard information about the other shape. In
future, we might support concatenation that preserves this
information for use with slicing.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another `TensorShape`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `TensorShape` whose dimensions are the concatenation of the
dimensions in `self` and `other`.
</td>
</tr>

</table>



<h3 id="is_compatible_with"><code>is_compatible_with</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_compatible_with(
    other
)
</code></pre>

Returns True iff `self` is compatible with `other`.

Two possibly-partially-defined shapes are compatible if there
exists a fully-defined shape that both shapes can represent. Thus,
compatibility allows the shape inference code to reason about
partially-defined shapes. For example:

* TensorShape(None) is compatible with all shapes.

* TensorShape([None, None]) is compatible with all two-dimensional
  shapes, such as TensorShape([32, 784]), and also TensorShape(None). It is
  not compatible with, for example, TensorShape([None]) or
  TensorShape([None, None, None]).

* TensorShape([32, None]) is compatible with all two-dimensional shapes
  with size 32 in the 0th dimension, and also TensorShape([None, None])
  and TensorShape(None). It is not compatible with, for example,
  TensorShape([32]), TensorShape([32, None, 1]) or TensorShape([64, None]).

* TensorShape([32, 784]) is compatible with itself, and also
  TensorShape([32, None]), TensorShape([None, 784]), TensorShape([None,
  None]) and TensorShape(None). It is not compatible with, for example,
  TensorShape([32, 1, 784]) or TensorShape([None]).

The compatibility relation is reflexive and symmetric, but not
transitive. For example, TensorShape([32, 784]) is compatible with
TensorShape(None), and TensorShape(None) is compatible with
TensorShape([4, 4]), but TensorShape([32, 784]) is not compatible with
TensorShape([4, 4]).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another TensorShape.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
True iff `self` is compatible with `other`.
</td>
</tr>

</table>



<h3 id="is_fully_defined"><code>is_fully_defined</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_fully_defined()
</code></pre>

Returns True iff `self` is fully defined in every dimension.


<h3 id="merge_with"><code>merge_with</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>merge_with(
    other
)
</code></pre>

Returns a `TensorShape` combining the information in `self` and `other`.

The dimensions in `self` and `other` are merged element-wise,
according to the rules below:

```python
Dimension(n).merge_with(Dimension(None)) == Dimension(n)
Dimension(None).merge_with(Dimension(n)) == Dimension(n)
Dimension(None).merge_with(Dimension(None)) == Dimension(None)
# raises ValueError for n != m
Dimension(n).merge_with(Dimension(m))
```
>> ts = tf.TensorShape([1,2])
>> ot1 = tf.TensorShape([1,2])
>> ts.merge_with(ot).as_list()
[1,2]

>> ot2 = tf.TensorShape([1,None])
>> ts.merge_with(ot2).as_list()
[1,2]

>> ot3 = tf.TensorShape([None, None])
>> ot3.merge_with(ot2).as_list()
[1, None]

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another `TensorShape`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `TensorShape` containing the combined information of `self` and
`other`.
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
If `self` and `other` are not compatible.
</td>
</tr>
</table>



<h3 id="most_specific_compatible_shape"><code>most_specific_compatible_shape</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>most_specific_compatible_shape(
    other
)
</code></pre>

Returns the most specific TensorShape compatible with `self` and `other`.

* TensorShape([None, 1]) is the most specific TensorShape compatible with
  both TensorShape([2, 1]) and TensorShape([5, 1]). Note that
  TensorShape(None) is also compatible with above mentioned TensorShapes.

* TensorShape([1, 2, 3]) is the most specific TensorShape compatible with
  both TensorShape([1, 2, 3]) and TensorShape([1, 2, 3]). There are more
  less specific TensorShapes compatible with above mentioned TensorShapes,
  e.g. TensorShape([1, 2, None]), TensorShape(None).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
Another `TensorShape`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `TensorShape` which is the most specific compatible shape of `self`
and `other`.
</td>
</tr>

</table>



<h3 id="num_elements"><code>num_elements</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>num_elements()
</code></pre>

Returns the total number of elements, or none for incomplete shapes.


<h3 id="with_rank"><code>with_rank</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_rank(
    rank
)
</code></pre>

Returns a shape based on `self` with the given rank.

This method promotes a completely unknown shape to one with a
known rank.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`rank`
</td>
<td>
An integer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A shape that is at least as specific as `self` with the given rank.
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
If `self` does not represent a shape with the given `rank`.
</td>
</tr>
</table>



<h3 id="with_rank_at_least"><code>with_rank_at_least</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_rank_at_least(
    rank
)
</code></pre>

Returns a shape based on `self` with at least the given rank.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`rank`
</td>
<td>
An integer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A shape that is at least as specific as `self` with at least the given
rank.
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
If `self` does not represent a shape with at least the given
`rank`.
</td>
</tr>
</table>



<h3 id="with_rank_at_most"><code>with_rank_at_most</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_rank_at_most(
    rank
)
</code></pre>

Returns a shape based on `self` with at most the given rank.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`rank`
</td>
<td>
An integer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A shape that is at least as specific as `self` with at most the given
rank.
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
If `self` does not represent a shape with at most the given
`rank`.
</td>
</tr>
</table>



<h3 id="__add__"><code>__add__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    other
)
</code></pre>




<h3 id="__bool__"><code>__bool__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__bool__()
</code></pre>

Returns True if this shape contains non-zero information.


<h3 id="__concat__"><code>__concat__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__concat__(
    other
)
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Returns True if `self` is equivalent to `other`.

It first tries to convert `other` to `TensorShape`. `TypeError` is thrown
when the conversion fails. Otherwise, it compares each element in the
TensorShape dimensions.

* Two *Fully known* shapes, return True iff each element is equal.
```
>>> t_a = tf.TensorShape([1,2])
>>> a = [1, 2]
>>> t_b = tf.TensorShape([1,2])
>>> t_c = tf.TensorShape([1,2,3])
>>> t_a.__eq__(a)
True
>>> t_a.__eq__(t_b)
True
>>> t_a.__eq__(t_c)
False
```

* Two *Partially-known* shapes, return False.
```
>>> p_a = tf.TensorShape([1,None])
>>> p_b = tf.TensorShape([2,None])
>>> p_a.__eq__(p_b)
False
>>> t_a.__eq__(p_a)
False
```

* Two *Unknown shape*, return True.
```
>>> unk_a = tf.TensorShape(None)
>>> unk_b = tf.TensorShape(None)
>>> unk_a.__eq__(unk_b)
True
>>> unk_a.__eq__(t_a)
False
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
A `TensorShape` or type that can be converted to `TensorShape`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
True if the dimensions are all equal.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>
<tr class="alt">
<td colspan="2">
TypeError if `other` can not be converted to `TensorShape`.
</td>
</tr>

</table>



<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    key
)
</code></pre>

Returns the value of a dimension or a shape, depending on the key.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`key`
</td>
<td>
If `key` is an integer, returns the dimension at that index;
otherwise if `key` is a slice, returns a TensorShape whose dimensions
are those selected by the slice from `self`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An integer if `key` is an integer, or a `TensorShape` if `key` is a
slice.
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
If `key` is a slice and `self` is completely unknown and
the step is set.
</td>
</tr>
</table>



<h3 id="__iter__"><code>__iter__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__()
</code></pre>

Returns `self.dims` if the rank is known, otherwise raises ValueError.


<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__()
</code></pre>

Returns the rank of this shape, or raises ValueError if unspecified.


<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Returns True if `self` is known to be different from `other`.


<h3 id="__nonzero__"><code>__nonzero__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__nonzero__()
</code></pre>

Returns True if this shape contains non-zero information.


<h3 id="__radd__"><code>__radd__</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/framework/tensor_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__radd__(
    other
)
</code></pre>






