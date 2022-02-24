description: Specification of target device used to optimize the model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.lite.TargetSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.lite.TargetSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/lite/python/lite.py">View source</a>



Specification of target device used to optimize the model.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.lite.TargetSpec`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.lite.TargetSpec(
    supported_ops=None, supported_types=None, experimental_select_user_tf_ops=None,
    experimental_supported_backends=None
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`supported_ops`
</td>
<td>
Experimental flag, subject to change. Set of <a href="../../tf/lite/OpsSet.md"><code>tf.lite.OpsSet</code></a>
options, where each option represents a set of operators supported by the
target device. (default {tf.lite.OpsSet.TFLITE_BUILTINS}))
</td>
</tr><tr>
<td>
`supported_types`
</td>
<td>
Set of <a href="../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a> data types supported on the target
device. If initialized, optimization might be driven by the smallest type
in this set. (default set())
</td>
</tr><tr>
<td>
`experimental_select_user_tf_ops`
</td>
<td>
Experimental flag, subject to change. Set
of user's TensorFlow operators' names that are required in the TensorFlow
Lite runtime. These ops will be exported as select TensorFlow ops in the
model (in conjunction with the tf.lite.OpsSet.SELECT_TF_OPS flag). This is
an advanced feature that should only be used if the client is using TF ops
that may not be linked in by default with the TF ops that are provided
when using the SELECT_TF_OPS path. The client is responsible for linking
these ops into the target runtime.
</td>
</tr><tr>
<td>
`experimental_supported_backends`
</td>
<td>
Experimental flag, subject to change.
Set containing names of supported backends. Currently only "GPU" is
supported, more options will be available later.
</td>
</tr>
</table>



