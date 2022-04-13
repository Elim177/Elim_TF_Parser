description: Adds a externally defined loss to the collection of losses.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.losses.add_loss" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.losses.add_loss

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/losses/util.py">View source</a>



Adds a externally defined loss to the collection of losses.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.losses.add_loss(
    loss, loss_collection=tf.GraphKeys.LOSSES
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`loss`
</td>
<td>
A loss `Tensor`.
</td>
</tr><tr>
<td>
`loss_collection`
</td>
<td>
Optional collection to add the loss to.
</td>
</tr>
</table>

