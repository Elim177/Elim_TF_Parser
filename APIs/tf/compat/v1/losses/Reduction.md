description: Types of loss reduction.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.losses.Reduction" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="all"/>
<meta itemprop="property" content="validate"/>
<meta itemprop="property" content="MEAN"/>
<meta itemprop="property" content="NONE"/>
<meta itemprop="property" content="SUM"/>
<meta itemprop="property" content="SUM_BY_NONZERO_WEIGHTS"/>
<meta itemprop="property" content="SUM_OVER_BATCH_SIZE"/>
<meta itemprop="property" content="SUM_OVER_NONZERO_WEIGHTS"/>
</div>

# tf.compat.v1.losses.Reduction

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/losses/losses_impl.py">View source</a>



Types of loss reduction.

<!-- Placeholder for "Used in" -->

Contains the following values:

* `NONE`: Un-reduced weighted losses with the same shape as input.
* `SUM`: Scalar sum of weighted losses.
* `MEAN`: Scalar `SUM` divided by sum of weights. DEPRECATED.
* `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.
* `SUM_OVER_NONZERO_WEIGHTS`: Scalar `SUM` divided by number of non-zero
   weights. DEPRECATED.
* `SUM_BY_NONZERO_WEIGHTS`: Same as `SUM_OVER_NONZERO_WEIGHTS`. DEPRECATED.

## Methods

<h3 id="all"><code>all</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/ops/losses/losses_impl.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>all()
</code></pre>




<h3 id="validate"><code>validate</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/ops/losses/losses_impl.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>validate(
    key
)
</code></pre>








<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
MEAN<a id="MEAN"></a>
</td>
<td>
`'weighted_mean'`
</td>
</tr><tr>
<td>
NONE<a id="NONE"></a>
</td>
<td>
`'none'`
</td>
</tr><tr>
<td>
SUM<a id="SUM"></a>
</td>
<td>
`'weighted_sum'`
</td>
</tr><tr>
<td>
SUM_BY_NONZERO_WEIGHTS<a id="SUM_BY_NONZERO_WEIGHTS"></a>
</td>
<td>
`'weighted_sum_by_nonzero_weights'`
</td>
</tr><tr>
<td>
SUM_OVER_BATCH_SIZE<a id="SUM_OVER_BATCH_SIZE"></a>
</td>
<td>
`'weighted_sum_over_batch_size'`
</td>
</tr><tr>
<td>
SUM_OVER_NONZERO_WEIGHTS<a id="SUM_OVER_NONZERO_WEIGHTS"></a>
</td>
<td>
`'weighted_sum_by_nonzero_weights'`
</td>
</tr>
</table>

