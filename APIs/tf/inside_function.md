description: Indicates whether the caller code is executing inside a <a href="../tf/function.md"><code>tf.function</code></a>.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.inside_function" />
<meta itemprop="path" content="Stable" />
</div>

# tf.inside_function

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>



Indicates whether the caller code is executing inside a <a href="../tf/function.md"><code>tf.function</code></a>.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.inside_function()
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Boolean, True if the caller code is executing inside a <a href="../tf/function.md"><code>tf.function</code></a>
rather than eagerly.
</td>
</tr>

</table>



#### Example:



```
>>> tf.inside_function()
False
>>> @tf.function
... def f():
...   print(tf.inside_function())
>>> f()
True
```