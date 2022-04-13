description: Public API for tf.saved_model namespace.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.saved_model" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="ASSETS_DIRECTORY"/>
<meta itemprop="property" content="ASSETS_KEY"/>
<meta itemprop="property" content="CLASSIFY_INPUTS"/>
<meta itemprop="property" content="CLASSIFY_METHOD_NAME"/>
<meta itemprop="property" content="CLASSIFY_OUTPUT_CLASSES"/>
<meta itemprop="property" content="CLASSIFY_OUTPUT_SCORES"/>
<meta itemprop="property" content="DEBUG_DIRECTORY"/>
<meta itemprop="property" content="DEBUG_INFO_FILENAME_PB"/>
<meta itemprop="property" content="DEFAULT_SERVING_SIGNATURE_DEF_KEY"/>
<meta itemprop="property" content="GPU"/>
<meta itemprop="property" content="LEGACY_INIT_OP_KEY"/>
<meta itemprop="property" content="MAIN_OP_KEY"/>
<meta itemprop="property" content="PREDICT_INPUTS"/>
<meta itemprop="property" content="PREDICT_METHOD_NAME"/>
<meta itemprop="property" content="PREDICT_OUTPUTS"/>
<meta itemprop="property" content="REGRESS_INPUTS"/>
<meta itemprop="property" content="REGRESS_METHOD_NAME"/>
<meta itemprop="property" content="REGRESS_OUTPUTS"/>
<meta itemprop="property" content="SAVED_MODEL_FILENAME_PB"/>
<meta itemprop="property" content="SAVED_MODEL_FILENAME_PBTXT"/>
<meta itemprop="property" content="SAVED_MODEL_SCHEMA_VERSION"/>
<meta itemprop="property" content="SERVING"/>
<meta itemprop="property" content="TPU"/>
<meta itemprop="property" content="TRAINING"/>
<meta itemprop="property" content="VARIABLES_DIRECTORY"/>
<meta itemprop="property" content="VARIABLES_FILENAME"/>
</div>

# Module: tf.compat.v1.saved_model

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Public API for tf.saved_model namespace.



## Modules

[`builder`](../../../tf/compat/v1/saved_model/builder.md) module: SavedModel builder.

[`constants`](../../../tf/compat/v1/saved_model/constants.md) module: Constants for SavedModel save and restore operations.

[`experimental`](../../../tf/compat/v1/saved_model/experimental.md) module: Public API for tf.saved_model.experimental namespace.

[`loader`](../../../tf/compat/v1/saved_model/loader.md) module: Loader functionality for SavedModel with hermetic, language-neutral exports.

[`main_op`](../../../tf/compat/v1/saved_model/main_op.md) module: SavedModel main op.

[`signature_constants`](../../../tf/compat/v1/saved_model/signature_constants.md) module: Signature constants for SavedModel save and restore operations.

[`signature_def_utils`](../../../tf/compat/v1/saved_model/signature_def_utils.md) module: SignatureDef utility functions.

[`tag_constants`](../../../tf/compat/v1/saved_model/tag_constants.md) module: Common tags used for graphs in SavedModel.

[`utils`](../../../tf/compat/v1/saved_model/utils.md) module: SavedModel utility functions.

## Classes

[`class Asset`](../../../tf/saved_model/Asset.md): Represents a file asset to hermetically include in a SavedModel.

[`class Builder`](../../../tf/compat/v1/saved_model/Builder.md): Builds the `SavedModel` protocol buffer and saves variables and assets.

[`class SaveOptions`](../../../tf/saved_model/SaveOptions.md): Options for saving to SavedModel.

## Functions

[`build_signature_def(...)`](../../../tf/compat/v1/saved_model/build_signature_def.md): Utility function to build a SignatureDef protocol buffer.

[`build_tensor_info(...)`](../../../tf/compat/v1/saved_model/build_tensor_info.md): Utility function to build TensorInfo proto from a Tensor. (deprecated)

[`classification_signature_def(...)`](../../../tf/compat/v1/saved_model/classification_signature_def.md): Creates classification signature from given examples and predictions.

[`contains_saved_model(...)`](../../../tf/compat/v1/saved_model/contains_saved_model.md): Checks whether the provided export directory could contain a SavedModel.

[`get_tensor_from_tensor_info(...)`](../../../tf/compat/v1/saved_model/get_tensor_from_tensor_info.md): Returns the Tensor or CompositeTensor described by a TensorInfo proto. (deprecated)

[`is_valid_signature(...)`](../../../tf/compat/v1/saved_model/is_valid_signature.md): Determine whether a SignatureDef can be served by TensorFlow Serving.

[`load(...)`](../../../tf/compat/v1/saved_model/load.md): Loads the model from a SavedModel as specified by tags. (deprecated)

[`load_v2(...)`](../../../tf/saved_model/load.md): Load a SavedModel from `export_dir`.

[`main_op_with_restore(...)`](../../../tf/compat/v1/saved_model/main_op_with_restore.md): Returns a main op to init variables, tables and restore the graph. (deprecated)

[`maybe_saved_model_directory(...)`](../../../tf/compat/v1/saved_model/contains_saved_model.md): Checks whether the provided export directory could contain a SavedModel.

[`predict_signature_def(...)`](../../../tf/compat/v1/saved_model/predict_signature_def.md): Creates prediction signature from given inputs and outputs.

[`regression_signature_def(...)`](../../../tf/compat/v1/saved_model/regression_signature_def.md): Creates regression signature from given examples and predictions.

[`save(...)`](../../../tf/saved_model/save.md): Exports a [tf.Module](https://www.tensorflow.org/api_docs/python/tf/Module) (and subclasses) `obj` to [SavedModel format](https://www.tensorflow.org/guide/saved_model#the_savedmodel_format_on_disk).

[`simple_save(...)`](../../../tf/compat/v1/saved_model/simple_save.md): Convenience function to build a SavedModel suitable for serving. (deprecated)



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
ASSETS_DIRECTORY<a id="ASSETS_DIRECTORY"></a>
</td>
<td>
`'assets'`
</td>
</tr><tr>
<td>
ASSETS_KEY<a id="ASSETS_KEY"></a>
</td>
<td>
`'saved_model_assets'`
</td>
</tr><tr>
<td>
CLASSIFY_INPUTS<a id="CLASSIFY_INPUTS"></a>
</td>
<td>
`'inputs'`
</td>
</tr><tr>
<td>
CLASSIFY_METHOD_NAME<a id="CLASSIFY_METHOD_NAME"></a>
</td>
<td>
`'tensorflow/serving/classify'`
</td>
</tr><tr>
<td>
CLASSIFY_OUTPUT_CLASSES<a id="CLASSIFY_OUTPUT_CLASSES"></a>
</td>
<td>
`'classes'`
</td>
</tr><tr>
<td>
CLASSIFY_OUTPUT_SCORES<a id="CLASSIFY_OUTPUT_SCORES"></a>
</td>
<td>
`'scores'`
</td>
</tr><tr>
<td>
DEBUG_DIRECTORY<a id="DEBUG_DIRECTORY"></a>
</td>
<td>
`'debug'`
</td>
</tr><tr>
<td>
DEBUG_INFO_FILENAME_PB<a id="DEBUG_INFO_FILENAME_PB"></a>
</td>
<td>
`'saved_model_debug_info.pb'`
</td>
</tr><tr>
<td>
DEFAULT_SERVING_SIGNATURE_DEF_KEY<a id="DEFAULT_SERVING_SIGNATURE_DEF_KEY"></a>
</td>
<td>
`'serving_default'`
</td>
</tr><tr>
<td>
GPU<a id="GPU"></a>
</td>
<td>
`'gpu'`
</td>
</tr><tr>
<td>
LEGACY_INIT_OP_KEY<a id="LEGACY_INIT_OP_KEY"></a>
</td>
<td>
`'legacy_init_op'`
</td>
</tr><tr>
<td>
MAIN_OP_KEY<a id="MAIN_OP_KEY"></a>
</td>
<td>
`'saved_model_main_op'`
</td>
</tr><tr>
<td>
PREDICT_INPUTS<a id="PREDICT_INPUTS"></a>
</td>
<td>
`'inputs'`
</td>
</tr><tr>
<td>
PREDICT_METHOD_NAME<a id="PREDICT_METHOD_NAME"></a>
</td>
<td>
`'tensorflow/serving/predict'`
</td>
</tr><tr>
<td>
PREDICT_OUTPUTS<a id="PREDICT_OUTPUTS"></a>
</td>
<td>
`'outputs'`
</td>
</tr><tr>
<td>
REGRESS_INPUTS<a id="REGRESS_INPUTS"></a>
</td>
<td>
`'inputs'`
</td>
</tr><tr>
<td>
REGRESS_METHOD_NAME<a id="REGRESS_METHOD_NAME"></a>
</td>
<td>
`'tensorflow/serving/regress'`
</td>
</tr><tr>
<td>
REGRESS_OUTPUTS<a id="REGRESS_OUTPUTS"></a>
</td>
<td>
`'outputs'`
</td>
</tr><tr>
<td>
SAVED_MODEL_FILENAME_PB<a id="SAVED_MODEL_FILENAME_PB"></a>
</td>
<td>
`'saved_model.pb'`
</td>
</tr><tr>
<td>
SAVED_MODEL_FILENAME_PBTXT<a id="SAVED_MODEL_FILENAME_PBTXT"></a>
</td>
<td>
`'saved_model.pbtxt'`
</td>
</tr><tr>
<td>
SAVED_MODEL_SCHEMA_VERSION<a id="SAVED_MODEL_SCHEMA_VERSION"></a>
</td>
<td>
`1`
</td>
</tr><tr>
<td>
SERVING<a id="SERVING"></a>
</td>
<td>
`'serve'`
</td>
</tr><tr>
<td>
TPU<a id="TPU"></a>
</td>
<td>
`'tpu'`
</td>
</tr><tr>
<td>
TRAINING<a id="TRAINING"></a>
</td>
<td>
`'train'`
</td>
</tr><tr>
<td>
VARIABLES_DIRECTORY<a id="VARIABLES_DIRECTORY"></a>
</td>
<td>
`'variables'`
</td>
</tr><tr>
<td>
VARIABLES_FILENAME<a id="VARIABLES_FILENAME"></a>
</td>
<td>
`'variables'`
</td>
</tr>
</table>

