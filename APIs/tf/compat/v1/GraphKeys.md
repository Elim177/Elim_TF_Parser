description: Standard names to use for graph collections.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.GraphKeys" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="ACTIVATIONS"/>
<meta itemprop="property" content="ASSET_FILEPATHS"/>
<meta itemprop="property" content="BIASES"/>
<meta itemprop="property" content="CONCATENATED_VARIABLES"/>
<meta itemprop="property" content="COND_CONTEXT"/>
<meta itemprop="property" content="EVAL_STEP"/>
<meta itemprop="property" content="GLOBAL_STEP"/>
<meta itemprop="property" content="GLOBAL_VARIABLES"/>
<meta itemprop="property" content="INIT_OP"/>
<meta itemprop="property" content="LOCAL_INIT_OP"/>
<meta itemprop="property" content="LOCAL_RESOURCES"/>
<meta itemprop="property" content="LOCAL_VARIABLES"/>
<meta itemprop="property" content="LOSSES"/>
<meta itemprop="property" content="METRIC_VARIABLES"/>
<meta itemprop="property" content="MODEL_VARIABLES"/>
<meta itemprop="property" content="MOVING_AVERAGE_VARIABLES"/>
<meta itemprop="property" content="QUEUE_RUNNERS"/>
<meta itemprop="property" content="READY_FOR_LOCAL_INIT_OP"/>
<meta itemprop="property" content="READY_OP"/>
<meta itemprop="property" content="REGULARIZATION_LOSSES"/>
<meta itemprop="property" content="RESOURCES"/>
<meta itemprop="property" content="SAVEABLE_OBJECTS"/>
<meta itemprop="property" content="SAVERS"/>
<meta itemprop="property" content="SUMMARIES"/>
<meta itemprop="property" content="SUMMARY_OP"/>
<meta itemprop="property" content="TABLE_INITIALIZERS"/>
<meta itemprop="property" content="TRAINABLE_RESOURCE_VARIABLES"/>
<meta itemprop="property" content="TRAINABLE_VARIABLES"/>
<meta itemprop="property" content="TRAIN_OP"/>
<meta itemprop="property" content="UPDATE_OPS"/>
<meta itemprop="property" content="VARIABLES"/>
<meta itemprop="property" content="WEIGHTS"/>
<meta itemprop="property" content="WHILE_CONTEXT"/>
</div>

# tf.compat.v1.GraphKeys

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>



Standard names to use for graph collections.

<!-- Placeholder for "Used in" -->

The standard library uses various well-known names to collect and
retrieve values associated with a graph. For example, the
`tf.Optimizer` subclasses default to optimizing the variables
collected under `tf.GraphKeys.TRAINABLE_VARIABLES` if none is
specified, but it is also possible to pass an explicit list of
variables.

The following standard keys are defined:

* `GLOBAL_VARIABLES`: the default collection of `Variable` objects, shared
  across distributed environment (model variables are subset of these). See
  <a href="../../../tf/compat/v1/global_variables.md"><code>tf.compat.v1.global_variables</code></a>
  for more details.
  Commonly, all `TRAINABLE_VARIABLES` variables will be in `MODEL_VARIABLES`,
  and all `MODEL_VARIABLES` variables will be in `GLOBAL_VARIABLES`.
* `LOCAL_VARIABLES`: the subset of `Variable` objects that are local to each
  machine. Usually used for temporarily variables, like counters.
  Note: use `tf.contrib.framework.local_variable` to add to this collection.
* `MODEL_VARIABLES`: the subset of `Variable` objects that are used in the
  model for inference (feed forward). Note: use
  `tf.contrib.framework.model_variable` to add to this collection.
* `TRAINABLE_VARIABLES`: the subset of `Variable` objects that will
  be trained by an optimizer. See
  <a href="../../../tf/compat/v1/trainable_variables.md"><code>tf.compat.v1.trainable_variables</code></a>
  for more details.
* `SUMMARIES`: the summary `Tensor` objects that have been created in the
  graph. See
  <a href="../../../tf/compat/v1/summary/merge_all.md"><code>tf.compat.v1.summary.merge_all</code></a>
  for more details.
* `QUEUE_RUNNERS`: the `QueueRunner` objects that are used to
  produce input for a computation. See
  <a href="../../../tf/compat/v1/train/start_queue_runners.md"><code>tf.compat.v1.train.start_queue_runners</code></a>
  for more details.
* `MOVING_AVERAGE_VARIABLES`: the subset of `Variable` objects that will also
  keep moving averages.  See
  <a href="../../../tf/compat/v1/moving_average_variables.md"><code>tf.compat.v1.moving_average_variables</code></a>
  for more details.
* `REGULARIZATION_LOSSES`: regularization losses collected during graph
  construction.

The following standard keys are _defined_, but their collections are **not**
automatically populated as many of the others are:

* `WEIGHTS`
* `BIASES`
* `ACTIVATIONS`



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
ACTIVATIONS<a id="ACTIVATIONS"></a>
</td>
<td>
`'activations'`
</td>
</tr><tr>
<td>
ASSET_FILEPATHS<a id="ASSET_FILEPATHS"></a>
</td>
<td>
`'asset_filepaths'`
</td>
</tr><tr>
<td>
BIASES<a id="BIASES"></a>
</td>
<td>
`'biases'`
</td>
</tr><tr>
<td>
CONCATENATED_VARIABLES<a id="CONCATENATED_VARIABLES"></a>
</td>
<td>
`'concatenated_variables'`
</td>
</tr><tr>
<td>
COND_CONTEXT<a id="COND_CONTEXT"></a>
</td>
<td>
`'cond_context'`
</td>
</tr><tr>
<td>
EVAL_STEP<a id="EVAL_STEP"></a>
</td>
<td>
`'eval_step'`
</td>
</tr><tr>
<td>
GLOBAL_STEP<a id="GLOBAL_STEP"></a>
</td>
<td>
`'global_step'`
</td>
</tr><tr>
<td>
GLOBAL_VARIABLES<a id="GLOBAL_VARIABLES"></a>
</td>
<td>
`'variables'`
</td>
</tr><tr>
<td>
INIT_OP<a id="INIT_OP"></a>
</td>
<td>
`'init_op'`
</td>
</tr><tr>
<td>
LOCAL_INIT_OP<a id="LOCAL_INIT_OP"></a>
</td>
<td>
`'local_init_op'`
</td>
</tr><tr>
<td>
LOCAL_RESOURCES<a id="LOCAL_RESOURCES"></a>
</td>
<td>
`'local_resources'`
</td>
</tr><tr>
<td>
LOCAL_VARIABLES<a id="LOCAL_VARIABLES"></a>
</td>
<td>
`'local_variables'`
</td>
</tr><tr>
<td>
LOSSES<a id="LOSSES"></a>
</td>
<td>
`'losses'`
</td>
</tr><tr>
<td>
METRIC_VARIABLES<a id="METRIC_VARIABLES"></a>
</td>
<td>
`'metric_variables'`
</td>
</tr><tr>
<td>
MODEL_VARIABLES<a id="MODEL_VARIABLES"></a>
</td>
<td>
`'model_variables'`
</td>
</tr><tr>
<td>
MOVING_AVERAGE_VARIABLES<a id="MOVING_AVERAGE_VARIABLES"></a>
</td>
<td>
`'moving_average_variables'`
</td>
</tr><tr>
<td>
QUEUE_RUNNERS<a id="QUEUE_RUNNERS"></a>
</td>
<td>
`'queue_runners'`
</td>
</tr><tr>
<td>
READY_FOR_LOCAL_INIT_OP<a id="READY_FOR_LOCAL_INIT_OP"></a>
</td>
<td>
`'ready_for_local_init_op'`
</td>
</tr><tr>
<td>
READY_OP<a id="READY_OP"></a>
</td>
<td>
`'ready_op'`
</td>
</tr><tr>
<td>
REGULARIZATION_LOSSES<a id="REGULARIZATION_LOSSES"></a>
</td>
<td>
`'regularization_losses'`
</td>
</tr><tr>
<td>
RESOURCES<a id="RESOURCES"></a>
</td>
<td>
`'resources'`
</td>
</tr><tr>
<td>
SAVEABLE_OBJECTS<a id="SAVEABLE_OBJECTS"></a>
</td>
<td>
`'saveable_objects'`
</td>
</tr><tr>
<td>
SAVERS<a id="SAVERS"></a>
</td>
<td>
`'savers'`
</td>
</tr><tr>
<td>
SUMMARIES<a id="SUMMARIES"></a>
</td>
<td>
`'summaries'`
</td>
</tr><tr>
<td>
SUMMARY_OP<a id="SUMMARY_OP"></a>
</td>
<td>
`'summary_op'`
</td>
</tr><tr>
<td>
TABLE_INITIALIZERS<a id="TABLE_INITIALIZERS"></a>
</td>
<td>
`'table_initializer'`
</td>
</tr><tr>
<td>
TRAINABLE_RESOURCE_VARIABLES<a id="TRAINABLE_RESOURCE_VARIABLES"></a>
</td>
<td>
`'trainable_resource_variables'`
</td>
</tr><tr>
<td>
TRAINABLE_VARIABLES<a id="TRAINABLE_VARIABLES"></a>
</td>
<td>
`'trainable_variables'`
</td>
</tr><tr>
<td>
TRAIN_OP<a id="TRAIN_OP"></a>
</td>
<td>
`'train_op'`
</td>
</tr><tr>
<td>
UPDATE_OPS<a id="UPDATE_OPS"></a>
</td>
<td>
`'update_ops'`
</td>
</tr><tr>
<td>
VARIABLES<a id="VARIABLES"></a>
</td>
<td>
`'variables'`
</td>
</tr><tr>
<td>
WEIGHTS<a id="WEIGHTS"></a>
</td>
<td>
`'weights'`
</td>
</tr><tr>
<td>
WHILE_CONTEXT<a id="WHILE_CONTEXT"></a>
</td>
<td>
`'while_context'`
</td>
</tr>
</table>

