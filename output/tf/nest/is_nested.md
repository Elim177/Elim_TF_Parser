description: Returns true if its input is a collections.abc.Sequence (except strings).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nest.is_nested" />
<meta itemprop="path" content="Stable" />
</div>

# tf.nest.is_nested

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/util/nest.py">View source</a>



Returns true if its input is a collections.abc.Sequence (except strings).

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.nest.is_nested`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.nest.is_nested(
    seq
)
</code></pre>



<!-- Placeholder for "Used in" -->

  ```
  >>> tf.nest.is_nested("1234")
  False
  ```

  ```
  >>> tf.nest.is_nested([1, 3, [4, 5]])
  True
  ```

  ```
  >>> tf.nest.is_nested(((7, 8), (5, 6)))
  True
  ```

  ```
  >>> tf.nest.is_nested([])
  True
  ```

  ```
  >>> tf.nest.is_nested({"a": 1, "b": 2})
  True
  ```

  ```
  >>> tf.nest.is_nested({"a": 1, "b": 2}.keys())
  True
  ```

  ```
  >>> tf.nest.is_nested({"a": 1, "b": 2}.values())
  True
  ```

  ```
  >>> tf.nest.is_nested({"a": 1, "b": 2}.items())
  True
  ```

  ```
  >>> tf.nest.is_nested(set([1, 2]))
  False
  ```

  ```
  >>> ones = tf.ones([2, 3])
  >>> tf.nest.is_nested(ones)
  False
  ```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`seq`
</td>
<td>
an input sequence.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
True if the sequence is a not a string and is a collections.abc.Sequence
or a dict.
</td>
</tr>

</table>

