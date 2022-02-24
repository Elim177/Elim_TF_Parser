description: Creates a tf.compat.v1.Session for a chief.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.train.ChiefSessionCreator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_session"/>
</div>

# tf.compat.v1.train.ChiefSessionCreator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/training/monitored_session.py">View source</a>



Creates a tf.compat.v1.Session for a chief.

Inherits From: [`SessionCreator`](../../../../tf/compat/v1/train/SessionCreator.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.train.ChiefSessionCreator(
    scaffold=None, master=&#x27;&#x27;, config=None, checkpoint_dir=None,
    checkpoint_filename_with_path=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`scaffold`
</td>
<td>
A `Scaffold` used for gathering or building supportive ops. If
not specified a default one is created. It's used to finalize the graph.
</td>
</tr><tr>
<td>
`master`
</td>
<td>
`String` representation of the TensorFlow master to use.
</td>
</tr><tr>
<td>
`config`
</td>
<td>
`ConfigProto` proto used to configure the session.
</td>
</tr><tr>
<td>
`checkpoint_dir`
</td>
<td>
A string.  Optional path to a directory where to restore
variables.
</td>
</tr><tr>
<td>
`checkpoint_filename_with_path`
</td>
<td>
Full file name path to the checkpoint file.
</td>
</tr>
</table>



## Methods

<h3 id="create_session"><code>create_session</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/training/monitored_session.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_session()
</code></pre>






