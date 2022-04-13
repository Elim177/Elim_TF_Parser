description: Creates a recurrent neural network specified by RNNCell cell. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.nn.dynamic_rnn" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.nn.dynamic_rnn

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/rnn.py">View source</a>



Creates a recurrent neural network specified by RNNCell `cell`. (deprecated)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.nn.dynamic_rnn(
    cell, inputs, sequence_length=None, initial_state=None, dtype=None,
    parallel_iterations=None, swap_memory=(False), time_major=(False), scope=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

<a href="../../../../tf/compat/v1/nn/dynamic_rnn.md"><code>tf.compat.v1.nn.dynamic_rnn</code></a> is not compatible with eager execution and
<a href="../../../../tf/function.md"><code>tf.function</code></a>. Please use <a href="../../../../tf/keras/layers/RNN.md"><code>tf.keras.layers.RNN</code></a> instead for TF2 migration.
Take LSTM as an example, you can instantiate a <a href="../../../../tf/keras/layers/RNN.md"><code>tf.keras.layers.RNN</code></a> layer
with <a href="../../../../tf/keras/layers/LSTMCell.md"><code>tf.keras.layers.LSTMCell</code></a>, or directly via <a href="../../../../tf/keras/layers/LSTM.md"><code>tf.keras.layers.LSTM</code></a>. Once
the keras layer is created, you can get the output and states by calling
the layer with input and states. Please refer to [this
guide](https://www.tensorflow.org/guide/keras/rnn) for more details about
Keras RNN. You can also find more details about the difference and comparison
between Keras RNN and TF compat v1 rnn in [this
document](https://github.com/tensorflow/community/blob/master/rfcs/20180920-unify-rnn-interface.md)

#### Structural Mapping to Native TF2

Before:

```python
# create 2 LSTMCells
rnn_layers = [tf.compat.v1.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]

# create a RNN cell composed sequentially of a number of RNNCells
multi_rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(rnn_layers)

# 'outputs' is a tensor of shape [batch_size, max_time, 256]
# 'state' is a N-tuple where N is the number of LSTMCells containing a
# tf.nn.rnn_cell.LSTMStateTuple for each cell
outputs, state = tf.compat.v1.nn.dynamic_rnn(cell=multi_rnn_cell,
                                             inputs=data,
                                             dtype=tf.float32)
```

After:

```python
# RNN layer can take a list of cells, which will then stack them together.
# By default, keras RNN will only return the last timestep output and will not
# return states. If you need whole time sequence output as well as the states,
# you can set `return_sequences` and `return_state` to True.
rnn_layer = tf.keras.layers.RNN([tf.keras.layers.LSTMCell(128),
                                 tf.keras.layers.LSTMCell(256)],
                                return_sequences=True,
                                return_state=True)
outputs, output_states = rnn_layer(inputs, states)
```

#### How to Map Arguments

| TF1 Arg Name          | TF2 Arg Name    | Note                             |
| :-------------------- | :-------------- | :------------------------------- |
| `cell`                | `cell`          | In the RNN layer constructor     |
| `inputs`              | `inputs`        | In the RNN layer `__call__`      |
| `sequence_length`     | Not used        | Adding masking layer before RNN  :
:                       :                 : to achieve the same result.      :
| `initial_state`       | `initial_state` | In the RNN layer `__call__`      |
| `dtype`               | `dtype`         | In the RNN layer constructor     |
| `parallel_iterations` | Not supported   |                                  |
| `swap_memory`         | Not supported   |                                  |
| `time_major`          | `time_major`    | In the RNN layer constructor     |
| `scope`               | Not supported   |                                  |


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use `keras.layers.RNN(cell)`, which is equivalent to this API

Performs fully dynamic unrolling of `inputs`.

#### Example:



```python
# create a BasicRNNCell
rnn_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(hidden_size)

# 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]

# defining initial state
initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

# 'state' is a tensor of shape [batch_size, cell_state_size]
outputs, state = tf.compat.v1.nn.dynamic_rnn(rnn_cell, input_data,
                                   initial_state=initial_state,
                                   dtype=tf.float32)
```

```python
# create 2 LSTMCells
rnn_layers = [tf.compat.v1.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]

# create a RNN cell composed sequentially of a number of RNNCells
multi_rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(rnn_layers)

# 'outputs' is a tensor of shape [batch_size, max_time, 256]
# 'state' is a N-tuple where N is the number of LSTMCells containing a
# tf.nn.rnn_cell.LSTMStateTuple for each cell
outputs, state = tf.compat.v1.nn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=data,
                                   dtype=tf.float32)
```


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`cell`
</td>
<td>
An instance of RNNCell.
</td>
</tr><tr>
<td>
`inputs`
</td>
<td>
The RNN inputs.
If `time_major == False` (default), this must be a `Tensor` of shape:
  `[batch_size, max_time, ...]`, or a nested tuple of such elements.
If `time_major == True`, this must be a `Tensor` of shape: `[max_time,
  batch_size, ...]`, or a nested tuple of such elements. This may also be
  a (possibly nested) tuple of Tensors satisfying this property.  The
  first two dimensions must match across all the inputs, but otherwise the
  ranks and other shape components may differ. In this case, input to
  `cell` at each time-step will replicate the structure of these tuples,
  except for the time dimension (from which the time is taken). The input
  to `cell` at each time step will be a `Tensor` or (possibly nested)
  tuple of Tensors each with dimensions `[batch_size, ...]`.
</td>
</tr><tr>
<td>
`sequence_length`
</td>
<td>
(optional) An int32/int64 vector sized `[batch_size]`. Used
to copy-through state and zero-out outputs when past a batch element's
sequence length.  This parameter enables users to extract the last valid
state and properly padded outputs, so it is provided for correctness.
</td>
</tr><tr>
<td>
`initial_state`
</td>
<td>
(optional) An initial state for the RNN. If `cell.state_size`
is an integer, this must be a `Tensor` of appropriate type and shape
`[batch_size, cell.state_size]`. If `cell.state_size` is a tuple, this
should be a tuple of tensors having shapes `[batch_size, s] for s in
cell.state_size`.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
(optional) The data type for the initial state and expected output.
Required if initial_state is not provided or RNN state has a heterogeneous
dtype.
</td>
</tr><tr>
<td>
`parallel_iterations`
</td>
<td>
(Default: 32).  The number of iterations to run in
parallel.  Those operations which do not have any temporal dependency and
can be run in parallel, will be.  This parameter trades off time for
space.  Values >> 1 use more memory but take less time, while smaller
values use less memory but computations take longer.
</td>
</tr><tr>
<td>
`swap_memory`
</td>
<td>
Transparently swap the tensors produced in forward inference
but needed for back prop from GPU to CPU.  This allows training RNNs which
would typically not fit on a single GPU, with very minimal (or no)
performance penalty.
</td>
</tr><tr>
<td>
`time_major`
</td>
<td>
The shape format of the `inputs` and `outputs` Tensors. If true,
these `Tensors` must be shaped `[max_time, batch_size, depth]`. If false,
these `Tensors` must be shaped `[batch_size, max_time, depth]`. Using
`time_major = True` is a bit more efficient because it avoids transposes
at the beginning and end of the RNN calculation.  However, most TensorFlow
data is batch-major, so by default this function accepts input and emits
output in batch-major form.
</td>
</tr><tr>
<td>
`scope`
</td>
<td>
VariableScope for the created subgraph; defaults to "rnn".
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A pair (outputs, state) where:
</td>
</tr>
<tr>
<td>
`outputs`
</td>
<td>
The RNN output `Tensor`.

If time_major == False (default), this will be a `Tensor` shaped:
  `[batch_size, max_time, cell.output_size]`.

If time_major == True, this will be a `Tensor` shaped:
  `[max_time, batch_size, cell.output_size]`.

Note, if `cell.output_size` is a (possibly nested) tuple of integers
or `TensorShape` objects, then `outputs` will be a tuple having the
same structure as `cell.output_size`, containing Tensors having shapes
corresponding to the shape data in `cell.output_size`.
</td>
</tr><tr>
<td>
`state`
</td>
<td>
The final state.  If `cell.state_size` is an int, this
will be shaped `[batch_size, cell.state_size]`.  If it is a
`TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
If it is a (possibly nested) tuple of ints or `TensorShape`, this will
be a tuple having the corresponding shapes. If cells are `LSTMCells`
`state` will be a tuple containing a `LSTMStateTuple` for each cell.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
If `cell` is not an instance of RNNCell.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If inputs is None or an empty list.
</td>
</tr>
</table>


