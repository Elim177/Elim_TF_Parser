description: Public API for tf.saved_model namespace.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.saved_model" />
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

# Module: tf.saved_model

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Public API for tf.saved_model namespace.



## Modules

[`experimental`](../tf/saved_model/experimental.md) module: Public API for tf.saved_model.experimental namespace.

## Classes

[`class Asset`](../tf/saved_model/Asset.md): Represents a file asset to hermetically include in a SavedModel.

[`class LoadOptions`](../tf/saved_model/LoadOptions.md): Options for loading a SavedModel.

[`class SaveOptions`](../tf/saved_model/SaveOptions.md): Options for saving to SavedModel.

## Functions

[`contains_saved_model(...)`](../tf/saved_model/contains_saved_model.md): Checks whether the provided export directory could contain a SavedModel.

[`load(...)`](../tf/saved_model/load.md): Load a SavedModel from `export_dir`.

[`save(...)`](../tf/saved_model/save.md): Exports a [tf.Module](https://www.tensorflow.org/api_docs/python/tf/Module) (and subclasses) `obj` to [SavedModel format](https://www.tensorflow.org/guide/saved_model#the_savedmodel_format_on_disk).



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

