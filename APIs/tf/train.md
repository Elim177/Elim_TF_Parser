description: Support for training models.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.train" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.train

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Support for training models.


See the [Training](https://tensorflow.org/api_guides/python/train) guide.

## Modules

[`experimental`](../tf/train/experimental.md) module: Public API for tf.train.experimental namespace.

## Classes

[`class BytesList`](../tf/train/BytesList.md): Container that holds repeated fundamental values of byte type in the <a href="../tf/train/Feature.md"><code>tf.train.Feature</code></a> message.

[`class Checkpoint`](../tf/train/Checkpoint.md): Manages saving/restoring trackable values to disk.

[`class CheckpointManager`](../tf/train/CheckpointManager.md): Manages multiple checkpoints by keeping some and deleting unneeded ones.

[`class CheckpointOptions`](../tf/train/CheckpointOptions.md): Options for constructing a Checkpoint.

[`class ClusterDef`](../tf/train/ClusterDef.md): A ProtocolMessage

[`class ClusterSpec`](../tf/train/ClusterSpec.md): Represents a cluster as a set of "tasks", organized into "jobs".

[`class Coordinator`](../tf/train/Coordinator.md): A coordinator for threads.

[`class Example`](../tf/train/Example.md): An `Example` is a mostly-normalized data format for storing data for training and inference.

[`class ExponentialMovingAverage`](../tf/train/ExponentialMovingAverage.md): Maintains moving averages of variables by employing an exponential decay.

[`class Feature`](../tf/train/Feature.md): A `Feature` is a list which may hold zero or more values.

[`class FeatureList`](../tf/train/FeatureList.md): Contains zero or more values of <a href="../tf/train/Feature.md"><code>tf.train.Feature</code></a>s.

[`class FeatureLists`](../tf/train/FeatureLists.md): Contains the mapping from name to <a href="../tf/train/FeatureList.md"><code>tf.train.FeatureList</code></a>.

[`class Features`](../tf/train/Features.md): Protocol message for describing the `features` of a <a href="../tf/train/Example.md"><code>tf.train.Example</code></a>.

[`class FloatList`](../tf/train/FloatList.md): Container that holds repeated fundamental values of float type in the <a href="../tf/train/Feature.md"><code>tf.train.Feature</code></a> message.

[`class Int64List`](../tf/train/Int64List.md): Container that holds repeated fundamental value of int64 type in the <a href="../tf/train/Feature.md"><code>tf.train.Feature</code></a> message.

[`class JobDef`](../tf/train/JobDef.md): A ProtocolMessage

[`class SequenceExample`](../tf/train/SequenceExample.md): A `SequenceExample` is a format for representing one or more sequences and some context.

[`class ServerDef`](../tf/train/ServerDef.md): A ProtocolMessage

## Functions

[`checkpoints_iterator(...)`](../tf/train/checkpoints_iterator.md): Continuously yield new checkpoint files as they appear.

[`get_checkpoint_state(...)`](../tf/train/get_checkpoint_state.md): Returns CheckpointState proto from the "checkpoint" file.

[`latest_checkpoint(...)`](../tf/train/latest_checkpoint.md): Finds the filename of latest saved checkpoint file.

[`list_variables(...)`](../tf/train/list_variables.md): Lists the checkpoint keys and shapes of variables in a checkpoint.

[`load_checkpoint(...)`](../tf/train/load_checkpoint.md): Returns `CheckpointReader` for checkpoint found in `ckpt_dir_or_file`.

[`load_variable(...)`](../tf/train/load_variable.md): Returns the tensor value of the given variable in the checkpoint.

