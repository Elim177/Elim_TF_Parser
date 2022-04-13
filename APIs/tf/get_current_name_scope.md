description: Returns current full name scope specified by <a href="../tf/name_scope.md"><code>tf.name_scope(...)</code></a>s.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.get_current_name_scope" />
<meta itemprop="path" content="Stable" />
</div>

# tf.get_current_name_scope

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>



Returns current full name scope specified by <a href="../tf/name_scope.md"><code>tf.name_scope(...)</code></a>s.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.get_current_name_scope()
</code></pre>



<!-- Placeholder for "Used in" -->

For example,
```python
with tf.name_scope("outer"):
  tf.get_current_name_scope()  # "outer"

  with tf.name_scope("inner"):
    tf.get_current_name_scope()  # "outer/inner"
```

In other words, <a href="../tf/get_current_name_scope.md"><code>tf.get_current_name_scope()</code></a> returns the op name prefix that
will be prepended to, if an op is created at that place.

Note that <a href="../tf/function.md"><code>@tf.function</code></a> resets the name scope stack as shown below.

```
with tf.name_scope("outer"):

  @tf.function
  def foo(x):
    with tf.name_scope("inner"):
      return tf.add(x * x)  # Op name is "inner/Add", not "outer/inner/Add"
```