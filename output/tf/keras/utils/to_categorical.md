description: Converts a class vector (integers) to binary class matrix.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.to_categorical" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.utils.to_categorical

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/utils/np_utils.py#L21-L74">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Converts a class vector (integers) to binary class matrix.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.utils.to_categorical`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.to_categorical(
    y, num_classes=None, dtype=&#x27;float32&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

E.g. for use with `categorical_crossentropy`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`y`
</td>
<td>
Array-like with class values to be converted into a matrix
(integers from 0 to `num_classes - 1`).
</td>
</tr><tr>
<td>
`num_classes`
</td>
<td>
Total number of classes. If `None`, this would be inferred
as `max(y) + 1`.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The data type expected by the input. Default: `'float32'`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A binary matrix representation of the input. The class axis is placed
last.
</td>
</tr>

</table>



#### Example:



```
>>> a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
>>> a = tf.constant(a, shape=[4, 4])
>>> print(a)
tf.Tensor(
  [[1. 0. 0. 0.]
   [0. 1. 0. 0.]
   [0. 0. 1. 0.]
   [0. 0. 0. 1.]], shape=(4, 4), dtype=float32)
```

```
>>> b = tf.constant([.9, .04, .03, .03,
...                  .3, .45, .15, .13,
...                  .04, .01, .94, .05,
...                  .12, .21, .5, .17],
...                 shape=[4, 4])
>>> loss = tf.keras.backend.categorical_crossentropy(a, b)
>>> print(np.around(loss, 5))
[0.10536 0.82807 0.1011  1.77196]
```

```
>>> loss = tf.keras.backend.categorical_crossentropy(a, a)
>>> print(np.around(loss, 5))
[0. 0. 0. 0.]
```