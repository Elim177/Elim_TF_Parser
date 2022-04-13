description: Public API for tf.experimental namespace.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.experimental

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Public API for tf.experimental namespace.



## Modules

[`dlpack`](../tf/experimental/dlpack.md) module: Public API for tf.experimental.dlpack namespace.

[`numpy`](../tf/experimental/numpy.md) module: # tf.experimental.numpy: NumPy API on TensorFlow.

[`tensorrt`](../tf/experimental/tensorrt.md) module: Public API for tf.experimental.tensorrt namespace.

## Classes

[`class BatchableExtensionType`](../tf/experimental/BatchableExtensionType.md): An ExtensionType that can be batched and unbatched.

[`class ExtensionType`](../tf/experimental/ExtensionType.md): Base class for TensorFlow `ExtensionType` classes.

[`class ExtensionTypeBatchEncoder`](../tf/experimental/ExtensionTypeBatchEncoder.md): Class used to encode and decode extension type values for batching.

[`class Optional`](../tf/experimental/Optional.md): Represents a value that may or may not be present.

## Functions

[`async_clear_error(...)`](../tf/experimental/async_clear_error.md): Clear pending operations and error statuses in async execution.

[`async_scope(...)`](../tf/experimental/async_scope.md): Context manager for grouping async operations.

[`dispatch_for_api(...)`](../tf/experimental/dispatch_for_api.md): Decorator that overrides the default implementation for a TensorFlow API.

[`dispatch_for_binary_elementwise_apis(...)`](../tf/experimental/dispatch_for_binary_elementwise_apis.md): Decorator to override default implementation for binary elementwise APIs.

[`dispatch_for_unary_elementwise_apis(...)`](../tf/experimental/dispatch_for_unary_elementwise_apis.md): Decorator to override default implementation for unary elementwise APIs.

[`function_executor_type(...)`](../tf/experimental/function_executor_type.md): Context manager for setting the executor of eager defined functions.

[`register_filesystem_plugin(...)`](../tf/experimental/register_filesystem_plugin.md): Loads a TensorFlow FileSystem plugin.

[`unregister_dispatch_for(...)`](../tf/experimental/unregister_dispatch_for.md): Unregisters a function that was registered with `@dispatch_for_*`.

