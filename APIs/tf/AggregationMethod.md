description: A class listing aggregation methods used to combine gradients.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.AggregationMethod" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="ADD_N"/>
<meta itemprop="property" content="DEFAULT"/>
<meta itemprop="property" content="EXPERIMENTAL_ACCUMULATE_N"/>
<meta itemprop="property" content="EXPERIMENTAL_TREE"/>
</div>

# tf.AggregationMethod

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/gradients_util.py">View source</a>



A class listing aggregation methods used to combine gradients.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.AggregationMethod`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

Computing partial derivatives can require aggregating gradient
contributions. This class lists the various methods that can
be used to combine gradients in the graph.

The following aggregation methods are part of the stable API for
aggregating gradients:

*  `ADD_N`: All of the gradient terms are summed as part of one
   operation using the "AddN" op (see <a href="../tf/math/add_n.md"><code>tf.add_n</code></a>). This
   method has the property that all gradients must be ready and
   buffered separately in memory before any aggregation is performed.
*  `DEFAULT`: The system-chosen default aggregation method.

The following aggregation methods are experimental and may not
be supported in future releases:

* `EXPERIMENTAL_TREE`: Gradient terms are summed in pairs using
  the "AddN" op. This method of summing gradients may reduce
  performance, but it can improve memory utilization because the
  gradients can be released earlier.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
ADD_N<a id="ADD_N"></a>
</td>
<td>
`0`
</td>
</tr><tr>
<td>
DEFAULT<a id="DEFAULT"></a>
</td>
<td>
`0`
</td>
</tr><tr>
<td>
EXPERIMENTAL_ACCUMULATE_N<a id="EXPERIMENTAL_ACCUMULATE_N"></a>
</td>
<td>
`2`
</td>
</tr><tr>
<td>
EXPERIMENTAL_TREE<a id="EXPERIMENTAL_TREE"></a>
</td>
<td>
`1`
</td>
</tr>
</table>

