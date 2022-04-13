description: Returns whether TensorFlow was built with XLA support.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.test.is_built_with_xla" />
<meta itemprop="path" content="Stable" />
</div>

# tf.test.is_built_with_xla

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/platform/test.py">View source</a>



Returns whether TensorFlow was built with XLA support.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.test.is_built_with_xla`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.test.is_built_with_xla()
</code></pre>



<!-- Placeholder for "Used in" -->

This method should only be used in tests written with <a href="../../tf/test/TestCase.md"><code>tf.test.TestCase</code></a>. A
typical usage is to skip tests that should only run with XLA.

```
>>> class MyTest(tf.test.TestCase):
...
...   def test_add_on_xla(self):
...     if not tf.test.is_built_with_xla():
...       self.skipTest("test is only applicable on XLA")
```

...     @tf.function(jit_compile=True)
...     def add(x, y):
...       return tf.math.add(x, y)
...
...     self.assertEqual(add(tf.ones(()), tf.ones(())), 2.0)

TensorFlow official binary is built with XLA.