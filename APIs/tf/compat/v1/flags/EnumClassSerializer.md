description: Class for generating string representations of an enum class flag value.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.EnumClassSerializer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="serialize"/>
</div>

# tf.compat.v1.flags.EnumClassSerializer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Class for generating string representations of an enum class flag value.

Inherits From: [`ArgumentSerializer`](../../../../tf/compat/v1/flags/ArgumentSerializer.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.app.flags.EnumClassSerializer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.EnumClassSerializer(
    lowercase
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`lowercase`
</td>
<td>
If True, enum member names are lowercased during serialization.
</td>
</tr>
</table>



## Methods

<h3 id="serialize"><code>serialize</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>serialize(
    value
)
</code></pre>

Returns a serialized string of the Enum class value.




