description: Enum defining options for variable handling when saving.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.saved_model.experimental.VariablePolicy" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="EXPAND_DISTRIBUTED_VARIABLES"/>
<meta itemprop="property" content="NONE"/>
<meta itemprop="property" content="SAVE_VARIABLE_DEVICES"/>
</div>

# tf.saved_model.experimental.VariablePolicy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/saved_model/save_options.py">View source</a>



Enum defining options for variable handling when saving.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.saved_model.experimental.VariablePolicy`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

NONE
  No policy applied: Distributed variables are saved as one variable, with no
  device attached.

SAVE_VARIABLE_DEVICES
  When saving variables, also save their device assignment.
  This is useful if one wants to hardcode devices in saved models, but it also
  makes them non-portable if soft device placement is disabled (more details
  in <a href="../../../tf/config/set_soft_device_placement.md"><code>tf.config.set_soft_device_placement</code></a>). This is currently not
  fully supported by <a href="../../../tf/saved_model/load.md"><code>saved_model.load</code></a>, and is mainly intended to be used
  when one will be reading the saved model at a lower API level. In the
  example below, the graph saved by the call to <a href="../../../tf/saved_model/save.md"><code>saved_model.save</code></a> will have
  the variable devices correctly specified:
  ```python
  exported = tf.train.Checkpoint()
  with tf.device('/GPU:0'):
    exported.x_gpu = tf.Variable(1.0)
  with tf.device('/CPU:0'):
    exported.x_cpu = tf.Variable(1.0)
  tf.saved_model.save(exported, export_dir,
      options = tf.saved_model.SaveOptions(
          experimental_variable_policy=
            tf.saved_model.experimental.VariablePolicy.SAVE_VARIABLE_DEVICES))
  ```
  Distributed variables are still saved as one variable under this policy.

EXPAND_DISTRIBUTED_VARIABLES
  Distributed variables will be saved with information about their components,
  allowing for their restoration on load. Also, the saved graph will contain
  references to those variables. This is useful when one wants to use the
  model for training in environments where the original distribution strategy
  is not available.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
EXPAND_DISTRIBUTED_VARIABLES<a id="EXPAND_DISTRIBUTED_VARIABLES"></a>
</td>
<td>
`<VariablePolicy.EXPAND_DISTRIBUTED_VARIABLES: 'expand_distributed_variables'>`
</td>
</tr><tr>
<td>
NONE<a id="NONE"></a>
</td>
<td>
`<VariablePolicy.NONE: None>`
</td>
</tr><tr>
<td>
SAVE_VARIABLE_DEVICES<a id="SAVE_VARIABLE_DEVICES"></a>
</td>
<td>
`<VariablePolicy.SAVE_VARIABLE_DEVICES: 'save_variable_devices'>`
</td>
</tr>
</table>

