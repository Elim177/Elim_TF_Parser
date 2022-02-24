description: Returns the Tensorflow eager tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dlpack.from_dlpack" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dlpack.from_dlpack

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/dlpack/dlpack.py">View source</a>



Returns the Tensorflow eager tensor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dlpack.from_dlpack(
    dlcapsule
)
</code></pre>



<!-- Placeholder for "Used in" -->

The returned tensor uses the memory shared by dlpack capsules from other
framework.

  ```python
  a = tf.experimental.dlpack.from_dlpack(dlcapsule)
  # `a` uses the memory shared by dlpack
  ```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dlcapsule`
</td>
<td>
A PyCapsule named as dltensor
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Tensorflow eager tensor
</td>
</tr>

</table>

