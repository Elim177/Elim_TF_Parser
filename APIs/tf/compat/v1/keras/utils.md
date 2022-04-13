description: Public API for tf.keras.utils namespace.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.keras.utils" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.compat.v1.keras.utils

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Public API for tf.keras.utils namespace.



## Classes

[`class CustomObjectScope`](../../../../tf/keras/utils/custom_object_scope.md): Exposes custom classes/functions to Keras deserialization internals.

[`class GeneratorEnqueuer`](../../../../tf/keras/utils/GeneratorEnqueuer.md): Builds a queue out of a data generator.

[`class OrderedEnqueuer`](../../../../tf/keras/utils/OrderedEnqueuer.md): Builds a Enqueuer from a Sequence.

[`class Progbar`](../../../../tf/keras/utils/Progbar.md): Displays a progress bar.

[`class Sequence`](../../../../tf/keras/utils/Sequence.md): Base object for fitting to a sequence of data, such as a dataset.

[`class SequenceEnqueuer`](../../../../tf/keras/utils/SequenceEnqueuer.md): Base class to enqueue inputs.

[`class custom_object_scope`](../../../../tf/keras/utils/custom_object_scope.md): Exposes custom classes/functions to Keras deserialization internals.

## Functions

[`array_to_img(...)`](../../../../tf/keras/utils/array_to_img.md): Converts a 3D Numpy array to a PIL Image instance.

[`deserialize_keras_object(...)`](../../../../tf/keras/utils/deserialize_keras_object.md): Turns the serialized form of a Keras object back into an actual object.

[`get_custom_objects(...)`](../../../../tf/keras/utils/get_custom_objects.md): Retrieves a live reference to the global dictionary of custom objects.

[`get_file(...)`](../../../../tf/keras/utils/get_file.md): Downloads a file from a URL if it not already in the cache.

[`get_registered_name(...)`](../../../../tf/keras/utils/get_registered_name.md): Returns the name registered to an object within the Keras framework.

[`get_registered_object(...)`](../../../../tf/keras/utils/get_registered_object.md): Returns the class associated with `name` if it is registered with Keras.

[`get_source_inputs(...)`](../../../../tf/keras/utils/get_source_inputs.md): Returns the list of input tensors necessary to compute `tensor`.

[`img_to_array(...)`](../../../../tf/keras/utils/img_to_array.md): Converts a PIL Image instance to a Numpy array.

[`load_img(...)`](../../../../tf/keras/utils/load_img.md): Loads an image into PIL format.

[`model_to_dot(...)`](../../../../tf/keras/utils/model_to_dot.md): Convert a Keras model to dot format.

[`normalize(...)`](../../../../tf/keras/utils/normalize.md): Normalizes a Numpy array.

[`plot_model(...)`](../../../../tf/keras/utils/plot_model.md): Converts a Keras model to dot format and save to a file.

[`register_keras_serializable(...)`](../../../../tf/keras/utils/register_keras_serializable.md): Registers an object with the Keras serialization framework.

[`save_img(...)`](../../../../tf/keras/utils/save_img.md): Saves an image stored as a Numpy array to a path or file object.

[`serialize_keras_object(...)`](../../../../tf/keras/utils/serialize_keras_object.md): Serialize a Keras object into a JSON-compatible representation.

[`to_categorical(...)`](../../../../tf/keras/utils/to_categorical.md): Converts a class vector (integers) to binary class matrix.

[`track_tf1_style_variables(...)`](../../../../tf/compat/v1/keras/utils/track_tf1_style_variables.md): Wrap layer & module methods in this decorator to capture tf1-style weights.

