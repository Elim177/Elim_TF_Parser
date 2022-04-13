description: Callback to back up and restore the training state.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.callbacks.experimental.BackupAndRestore" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="set_model"/>
<meta itemprop="property" content="set_params"/>
</div>

# tf.keras.callbacks.experimental.BackupAndRestore

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/callbacks.py#L1586-L1706">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Callback to back up and restore the training state.

Inherits From: [`Callback`](../../../../tf/keras/callbacks/Callback.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.callbacks.experimental.BackupAndRestore(
    backup_dir
)
</code></pre>



<!-- Placeholder for "Used in" -->

`BackupAndRestore` callback is intended to recover training from an
interruption that has happened in the middle of a <a href="../../../../tf/keras/Model.md#fit"><code>Model.fit</code></a> execution, by
backing up the training states in a temporary checkpoint file (with the help
of a <a href="../../../../tf/train/CheckpointManager.md"><code>tf.train.CheckpointManager</code></a>), at the end of each epoch. Each backup
overwrites the previously written checkpoint file, so at any given time there
is at most one such checkpoint file for backup/restoring purpose.

If training restarts before completion, the training state (which includes the
`Model` weights and epoch number) is restored to the most recently saved state
at the beginning of a new <a href="../../../../tf/keras/Model.md#fit"><code>Model.fit</code></a> run. At the completion of a <a href="../../../../tf/keras/Model.md#fit"><code>Model.fit</code></a>
run, the temporary checkpoint file is deleted.

Note that the user is responsible to bring jobs back after the interruption.
This callback is important for the backup and restore mechanism for fault
tolerance purpose, and the model to be restored from an previous checkpoint is
expected to be the same as the one used to back up. If user changes arguments
passed to compile or fit, the checkpoint saved for fault tolerance can become
invalid.

#### Note:


1. This callback is not compatible with eager execution disabled.
2. A checkpoint is saved at the end of each epoch. After restoring,
<a href="../../../../tf/keras/Model.md#fit"><code>Model.fit</code></a> redoes any partial work during the unfinished epoch in which the
training got restarted (so the work done before the interruption doesn't
affect the final model state).
3. This works for both single worker and multi-worker modes. When <a href="../../../../tf/keras/Model.md#fit"><code>Model.fit</code></a>
is used with <a href="../../../../tf/distribute.md"><code>tf.distribute</code></a>, it supports <a href="../../../../tf/distribute/MirroredStrategy.md"><code>tf.distribute.MirroredStrategy</code></a>,
<a href="../../../../tf/distribute/MultiWorkerMirroredStrategy.md"><code>tf.distribute.MultiWorkerMirroredStrategy</code></a>, <a href="../../../../tf/distribute/TPUStrategy.md"><code>tf.distribute.TPUStrategy</code></a>, and
<a href="../../../../tf/distribute/experimental/ParameterServerStrategy.md"><code>tf.distribute.experimental.ParameterServerStrategy</code></a>.

#### Example:



```
>>> class InterruptingCallback(tf.keras.callbacks.Callback):
...   def on_epoch_begin(self, epoch, logs=None):
...     if epoch == 4:
...       raise RuntimeError('Interrupting!')
>>> callback = tf.keras.callbacks.experimental.BackupAndRestore(
... backup_dir="/tmp/backup")
>>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
>>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
>>> try:
...   model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10,
...             batch_size=1, callbacks=[callback, InterruptingCallback()],
...             verbose=0)
... except:
...   pass
>>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10,
...             batch_size=1, callbacks=[callback], verbose=0)
>>> # Only 6 more epochs are run, since first trainning got interrupted at
>>> # zero-indexed epoch 4, second training will continue from 4 to 9.
>>> len(history.history['loss'])
6
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`backup_dir`
</td>
<td>
String, path to store the checkpoint.
e.g. backup_dir = os.path.join(working_dir, 'backup')
This is the directory in which the system stores temporary files to
recover the model from jobs terminated unexpectedly. The directory
cannot be reused elsewhere to store other files, e.g. by
BackupAndRestore callback of another training, or by another callback
(ModelCheckpoint) of the same training.
</td>
</tr>
</table>



## Methods

<h3 id="set_model"><code>set_model</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/callbacks.py#L645-L646">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_model(
    model
)
</code></pre>




<h3 id="set_params"><code>set_params</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/callbacks.py#L642-L643">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_params(
    params
)
</code></pre>






