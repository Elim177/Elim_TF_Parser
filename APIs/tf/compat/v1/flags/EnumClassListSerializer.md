description: A serializer for MultiEnumClass flags.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.EnumClassListSerializer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="serialize"/>
</div>

# tf.compat.v1.flags.EnumClassListSerializer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



A serializer for MultiEnumClass flags.

Inherits From: [`ListSerializer`](../../../../tf/compat/v1/flags/ListSerializer.md), [`ArgumentSerializer`](../../../../tf/compat/v1/flags/ArgumentSerializer.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.app.flags.EnumClassListSerializer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.EnumClassListSerializer(
    list_sep, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This serializer simply joins the output of `EnumClassSerializer` using a
provided seperator.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`list_sep`
</td>
<td>
String to be used as a separator when serializing
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword arguments to the `EnumClassSerializer` used to serialize
individual values.
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

See base class.




