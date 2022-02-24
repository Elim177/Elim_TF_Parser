description: Decorator that overrides the default implementation for a TensorFlow API.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dispatch_for_api" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dispatch_for_api

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/util/dispatch.py">View source</a>



Decorator that overrides the default implementation for a TensorFlow API.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.experimental.dispatch_for_api`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dispatch_for_api(
    api, *signatures
)
</code></pre>



<!-- Placeholder for "Used in" -->

The decorated function (known as the "dispatch target") will override the
default implementation for the API when the API is called with parameters that
match a specified type signature.  Signatures are specified using dictionaries
that map parameter names to type annotations.  E.g., in the following example,
`masked_add` will be called for <a href="../../tf/math/add.md"><code>tf.add</code></a> if both `x` and `y` are
`MaskedTensor`s:

```
>>> class MaskedTensor(extension_type.ExtensionType):
...   values: tf.Tensor
...   mask: tf.Tensor
```

```
>>> @dispatch_for_api(tf.math.add, {'x': MaskedTensor, 'y': MaskedTensor})
... def masked_add(x, y, name=None):
...   return MaskedTensor(x.values + y.values, x.mask & y.mask)
```

```
>>> mt = tf.add(MaskedTensor([1, 2], [True, False]), MaskedTensor(10, True))
>>> print(f"values={mt.values.numpy()}, mask={mt.mask.numpy()}")
values=[11 12], mask=[ True False]
```

If multiple type signatures are specified, then the dispatch target will be
called if any of the signatures match.  For example, the following code
registers `masked_add` to be called if `x` is a `MaskedTensor` *or* `y` is
a `MaskedTensor`.

```
>>> @dispatch_for_api(tf.math.add, {'x': MaskedTensor}, {'y':MaskedTensor})
... def masked_add(x, y):
...   x_values = x.values if isinstance(x, MaskedTensor) else x
...   x_mask = x.mask if isinstance(x, MaskedTensor) else True
...   y_values = y.values if isinstance(y, MaskedTensor) else y
...   y_mask = y.mask if isinstance(y, MaskedTensor) else True
...   return MaskedTensor(x_values + y_values, x_mask & y_mask)
```

The type annotations in type signatures may be type objects (e.g.,
`MaskedTensor`), `typing.List` values, or `typing.Union` values.   For
example, the following will register `masked_concat` to be called if `values`
is a list of `MaskedTensor` values:

```
>>> @dispatch_for_api(tf.concat, {'values': typing.List[MaskedTensor]})
... def masked_concat(values, axis):
...   return MaskedTensor(tf.concat([v.values for v in values], axis),
...                       tf.concat([v.mask for v in values], axis))
```

Each type signature must contain at least one subclass of `tf.CompositeTensor`
(which includes subclasses of `tf.ExtensionType`), and dispatch will only be
triggered if at least one type-annotated parameter contains a
`CompositeTensor` value.  This rule avoids invoking dispatch in degenerate
cases, such as the following examples:

* `@dispatch_for_api(tf.concat, {'values': List[MaskedTensor]})`: Will not
  dispatch to the decorated dispatch target when the user calls
  `tf.concat([])`.

* `@dispatch_for_api(tf.add, {'x': Union[MaskedTensor, Tensor], 'y':
  Union[MaskedTensor, Tensor]})`: Will not dispatch to the decorated dispatch
  target when the user calls `tf.add(tf.constant(1), tf.constant(2))`.

The dispatch target's signature must match the signature of the API that is
being overridden.  In particular, parameters must have the same names, and
must occur in the same order.  The dispatch target may optionally elide the
"name" parameter, in which case it will be wrapped with a call to
<a href="../../tf/name_scope.md"><code>tf.name_scope</code></a> when appropraite.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`api`
</td>
<td>
The TensorFlow API to override.
</td>
</tr><tr>
<td>
`*signatures`
</td>
<td>
Dictionaries mapping parameter names or indices to type
annotations, specifying when the dispatch target should be called.  In
particular, the dispatch target will be called if any signature matches;
and a signature matches if all of the specified parameters have types that
match with the indicated type annotations.  If no signatures are
specified, then a signature will be read from the dispatch target
function's type annotations.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A decorator that overrides the default implementation for `api`.
</td>
</tr>

</table>


#### Registered APIs

The TensorFlow APIs that may be overridden by `@dispatch_for_api` are:

* `tf.__operators__.add(x, y, name=None)`
* `tf.__operators__.eq(self, other)`
* `tf.__operators__.getitem(tensor, slice_spec, var=None)`
* `tf.__operators__.ne(self, other)`
* `tf.__operators__.ragged_getitem(rt_input, key)`
* `tf.argsort(values, axis=-1, direction='ASCENDING', stable=False, name=None)`
* `tf.audio.decode_wav(contents, desired_channels=-1, desired_samples=-1, name=None)`
* <a href="../../tf/audio/encode_wav.md"><code>tf.audio.encode_wav(audio, sample_rate, name=None)</code></a>
* <a href="../../tf/batch_to_space.md"><code>tf.batch_to_space(input, block_shape, crops, name=None)</code></a>
* <a href="../../tf/bitcast.md"><code>tf.bitcast(input, type, name=None)</code></a>
* <a href="../../tf/bitwise/bitwise_and.md"><code>tf.bitwise.bitwise_and(x, y, name=None)</code></a>
* <a href="../../tf/bitwise/bitwise_or.md"><code>tf.bitwise.bitwise_or(x, y, name=None)</code></a>
* <a href="../../tf/bitwise/bitwise_xor.md"><code>tf.bitwise.bitwise_xor(x, y, name=None)</code></a>
* <a href="../../tf/bitwise/invert.md"><code>tf.bitwise.invert(x, name=None)</code></a>
* <a href="../../tf/bitwise/left_shift.md"><code>tf.bitwise.left_shift(x, y, name=None)</code></a>
* <a href="../../tf/bitwise/right_shift.md"><code>tf.bitwise.right_shift(x, y, name=None)</code></a>
* `tf.boolean_mask(tensor, mask, axis=None, name='boolean_mask')`
* <a href="../../tf/broadcast_dynamic_shape.md"><code>tf.broadcast_dynamic_shape(shape_x, shape_y)</code></a>
* <a href="../../tf/broadcast_static_shape.md"><code>tf.broadcast_static_shape(shape_x, shape_y)</code></a>
* <a href="../../tf/broadcast_to.md"><code>tf.broadcast_to(input, shape, name=None)</code></a>
* `tf.case(pred_fn_pairs, default=None, exclusive=False, strict=False, name='case')`
* <a href="../../tf/cast.md"><code>tf.cast(x, dtype, name=None)</code></a>
* <a href="../../tf/clip_by_global_norm.md"><code>tf.clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None)</code></a>
* <a href="../../tf/clip_by_norm.md"><code>tf.clip_by_norm(t, clip_norm, axes=None, name=None)</code></a>
* <a href="../../tf/clip_by_value.md"><code>tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)</code></a>
* <a href="../../tf/compat/v1/Print.md"><code>tf.compat.v1.Print(input_, data, message=None, first_n=None, summarize=None, name=None)</code></a>
* <a href="../../tf/compat/v1/arg_max.md"><code>tf.compat.v1.arg_max(input, dimension, output_type=tf.int64, name=None)</code></a>
* <a href="../../tf/compat/v1/arg_min.md"><code>tf.compat.v1.arg_min(input, dimension, output_type=tf.int64, name=None)</code></a>
* <a href="../../tf/compat/v1/batch_gather.md"><code>tf.compat.v1.batch_gather(params, indices, name=None)</code></a>
* <a href="../../tf/compat/v1/batch_to_space.md"><code>tf.compat.v1.batch_to_space(input, crops, block_size, name=None, block_shape=None)</code></a>
* <a href="../../tf/compat/v1/batch_to_space_nd.md"><code>tf.compat.v1.batch_to_space_nd(input, block_shape, crops, name=None)</code></a>
* `tf.compat.v1.boolean_mask(tensor, mask, name='boolean_mask', axis=None)`
* `tf.compat.v1.case(pred_fn_pairs, default=None, exclusive=False, strict=False, name='case')`
* <a href="../../tf/compat/v1/clip_by_average_norm.md"><code>tf.compat.v1.clip_by_average_norm(t, clip_norm, name=None)</code></a>
* <a href="../../tf/compat/v1/cond.md"><code>tf.compat.v1.cond(pred, true_fn=None, false_fn=None, strict=False, name=None, fn1=None, fn2=None)</code></a>
* <a href="../../tf/compat/v1/convert_to_tensor.md"><code>tf.compat.v1.convert_to_tensor(value, dtype=None, name=None, preferred_dtype=None, dtype_hint=None)</code></a>
* <a href="../../tf/compat/v1/verify_tensor_all_finite.md"><code>tf.compat.v1.debugging.assert_all_finite(t=None, msg=None, name=None, x=None, message=None)</code></a>
* <a href="../../tf/compat/v1/assert_equal.md"><code>tf.compat.v1.debugging.assert_equal(x, y, data=None, summarize=None, message=None, name=None)</code></a>
* <a href="../../tf/compat/v1/assert_greater.md"><code>tf.compat.v1.debugging.assert_greater(x, y, data=None, summarize=None, message=None, name=None)</code></a>
* <a href="../../tf/compat/v1/assert_greater_equal.md"><code>tf.compat.v1.debugging.assert_greater_equal(x, y, data=None, summarize=None, message=None, name=None)</code></a>
* <a href="../../tf/compat/v1/assert_integer.md"><code>tf.compat.v1.debugging.assert_integer(x, message=None, name=None)</code></a>
* <a href="../../tf/compat/v1/assert_less.md"><code>tf.compat.v1.debugging.assert_less(x, y, data=None, summarize=None, message=None, name=None)</code></a>
* <a href="../../tf/compat/v1/assert_less_equal.md"><code>tf.compat.v1.debugging.assert_less_equal(x, y, data=None, summarize=None, message=None, name=None)</code></a>
* <a href="../../tf/compat/v1/assert_near.md"><code>tf.compat.v1.debugging.assert_near(x, y, rtol=None, atol=None, data=None, summarize=None, message=None, name=None)</code></a>
* <a href="../../tf/compat/v1/assert_negative.md"><code>tf.compat.v1.debugging.assert_negative(x, data=None, summarize=None, message=None, name=None)</code></a>
* <a href="../../tf/compat/v1/assert_non_negative.md"><code>tf.compat.v1.debugging.assert_non_negative(x, data=None, summarize=None, message=None, name=None)</code></a>
* <a href="../../tf/compat/v1/assert_non_positive.md"><code>tf.compat.v1.debugging.assert_non_positive(x, data=None, summarize=None, message=None, name=None)</code></a>
* <a href="../../tf/compat/v1/assert_none_equal.md"><code>tf.compat.v1.debugging.assert_none_equal(x, y, data=None, summarize=None, message=None, name=None)</code></a>
* <a href="../../tf/compat/v1/assert_positive.md"><code>tf.compat.v1.debugging.assert_positive(x, data=None, summarize=None, message=None, name=None)</code></a>
* <a href="../../tf/compat/v1/assert_rank.md"><code>tf.compat.v1.debugging.assert_rank(x, rank, data=None, summarize=None, message=None, name=None)</code></a>
* <a href="../../tf/compat/v1/assert_rank_at_least.md"><code>tf.compat.v1.debugging.assert_rank_at_least(x, rank, data=None, summarize=None, message=None, name=None)</code></a>
* <a href="../../tf/compat/v1/assert_rank_in.md"><code>tf.compat.v1.debugging.assert_rank_in(x, ranks, data=None, summarize=None, message=None, name=None)</code></a>
* <a href="../../tf/compat/v1/assert_scalar.md"><code>tf.compat.v1.debugging.assert_scalar(tensor, name=None, message=None)</code></a>
* <a href="../../tf/compat/v1/debugging/assert_shapes.md"><code>tf.compat.v1.debugging.assert_shapes(shapes, data=None, summarize=None, message=None, name=None)</code></a>
* <a href="../../tf/compat/v1/assert_type.md"><code>tf.compat.v1.debugging.assert_type(tensor, tf_type, message=None, name=None)</code></a>
* <a href="../../tf/compat/v1/decode_raw.md"><code>tf.compat.v1.decode_raw(input_bytes=None, out_type=None, little_endian=True, name=None, bytes=None)</code></a>
* <a href="../../tf/RaggedTensor.md#__div__"><code>tf.compat.v1.div(x, y, name=None)</code></a>
* <a href="../../tf/compat/v1/expand_dims.md"><code>tf.compat.v1.expand_dims(input, axis=None, name=None, dim=None)</code></a>
* <a href="../../tf/compat/v1/floor_div.md"><code>tf.compat.v1.floor_div(x, y, name=None)</code></a>
* <a href="../../tf/compat/v1/foldl.md"><code>tf.compat.v1.foldl(fn, elems, initializer=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)</code></a>
* <a href="../../tf/compat/v1/foldr.md"><code>tf.compat.v1.foldr(fn, elems, initializer=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)</code></a>
* <a href="../../tf/compat/v1/gather.md"><code>tf.compat.v1.gather(params, indices, validate_indices=None, name=None, axis=None, batch_dims=0)</code></a>
* <a href="../../tf/compat/v1/gather_nd.md"><code>tf.compat.v1.gather_nd(params, indices, name=None, batch_dims=0)</code></a>
* `tf.compat.v1.image.crop_and_resize(image, boxes, box_ind=None, crop_size=None, method='bilinear', extrapolation_value=0, name=None, box_indices=None)`
* <a href="../../tf/compat/v1/image/draw_bounding_boxes.md"><code>tf.compat.v1.image.draw_bounding_boxes(images, boxes, name=None, colors=None)</code></a>
* <a href="../../tf/compat/v1/image/extract_glimpse.md"><code>tf.compat.v1.image.extract_glimpse(input, size, offsets, centered=True, normalized=True, uniform_noise=True, name=None)</code></a>
* <a href="../../tf/compat/v1/extract_image_patches.md"><code>tf.compat.v1.image.extract_image_patches(images, ksizes=None, strides=None, rates=None, padding=None, name=None, sizes=None)</code></a>
* <a href="../../tf/compat/v1/image/resize_area.md"><code>tf.compat.v1.image.resize_area(images, size, align_corners=False, name=None)</code></a>
* <a href="../../tf/compat/v1/image/resize_bicubic.md"><code>tf.compat.v1.image.resize_bicubic(images, size, align_corners=False, name=None, half_pixel_centers=False)</code></a>
* <a href="../../tf/compat/v1/image/resize_bilinear.md"><code>tf.compat.v1.image.resize_bilinear(images, size, align_corners=False, name=None, half_pixel_centers=False)</code></a>
* <a href="../../tf/compat/v1/image/resize_image_with_pad.md"><code>tf.compat.v1.image.resize_image_with_pad(image, target_height, target_width, method=0, align_corners=False)</code></a>
* <a href="../../tf/compat/v1/image/resize.md"><code>tf.compat.v1.image.resize_images(images, size, method=0, align_corners=False, preserve_aspect_ratio=False, name=None)</code></a>
* <a href="../../tf/compat/v1/image/resize_nearest_neighbor.md"><code>tf.compat.v1.image.resize_nearest_neighbor(images, size, align_corners=False, name=None, half_pixel_centers=False)</code></a>
* <a href="../../tf/compat/v1/image/sample_distorted_bounding_box.md"><code>tf.compat.v1.image.sample_distorted_bounding_box(image_size, bounding_boxes, seed=None, seed2=None, min_object_covered=0.1, aspect_ratio_range=None, area_range=None, max_attempts=None, use_image_if_no_bounding_boxes=None, name=None)</code></a>
* `tf.compat.v1.io.decode_csv(records, record_defaults, field_delim=',', use_quote_delim=True, name=None, na_value='', select_cols=None)`
* <a href="../../tf/compat/v1/parse_example.md"><code>tf.compat.v1.io.parse_example(serialized, features, name=None, example_names=None)</code></a>
* <a href="../../tf/compat/v1/parse_single_example.md"><code>tf.compat.v1.io.parse_single_example(serialized, features, name=None, example_names=None)</code></a>
* <a href="../../tf/compat/v1/serialize_many_sparse.md"><code>tf.compat.v1.io.serialize_many_sparse(sp_input, name=None, out_type=tf.string)</code></a>
* <a href="../../tf/compat/v1/serialize_sparse.md"><code>tf.compat.v1.io.serialize_sparse(sp_input, name=None, out_type=tf.string)</code></a>
* `tf.compat.v1.losses.absolute_difference(labels, predictions, weights=1.0, scope=None, loss_collection='losses', reduction='weighted_sum_by_nonzero_weights')`
* `tf.compat.v1.losses.compute_weighted_loss(losses, weights=1.0, scope=None, loss_collection='losses', reduction='weighted_sum_by_nonzero_weights')`
* `tf.compat.v1.losses.cosine_distance(labels, predictions, axis=None, weights=1.0, scope=None, loss_collection='losses', reduction='weighted_sum_by_nonzero_weights', dim=None)`
* `tf.compat.v1.losses.hinge_loss(labels, logits, weights=1.0, scope=None, loss_collection='losses', reduction='weighted_sum_by_nonzero_weights')`
* `tf.compat.v1.losses.huber_loss(labels, predictions, weights=1.0, delta=1.0, scope=None, loss_collection='losses', reduction='weighted_sum_by_nonzero_weights')`
* `tf.compat.v1.losses.log_loss(labels, predictions, weights=1.0, epsilon=1e-07, scope=None, loss_collection='losses', reduction='weighted_sum_by_nonzero_weights')`
* `tf.compat.v1.losses.mean_pairwise_squared_error(labels, predictions, weights=1.0, scope=None, loss_collection='losses')`
* `tf.compat.v1.losses.mean_squared_error(labels, predictions, weights=1.0, scope=None, loss_collection='losses', reduction='weighted_sum_by_nonzero_weights')`
* `tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels, logits, weights=1.0, label_smoothing=0, scope=None, loss_collection='losses', reduction='weighted_sum_by_nonzero_weights')`
* `tf.compat.v1.losses.softmax_cross_entropy(onehot_labels, logits, weights=1.0, label_smoothing=0, scope=None, loss_collection='losses', reduction='weighted_sum_by_nonzero_weights')`
* `tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, logits, weights=1.0, scope=None, loss_collection='losses', reduction='weighted_sum_by_nonzero_weights')`
* <a href="../../tf/compat/v1/argmax.md"><code>tf.compat.v1.math.argmax(input, axis=None, name=None, dimension=None, output_type=tf.int64)</code></a>
* <a href="../../tf/compat/v1/argmin.md"><code>tf.compat.v1.math.argmin(input, axis=None, name=None, dimension=None, output_type=tf.int64)</code></a>
* <a href="../../tf/compat/v1/confusion_matrix.md"><code>tf.compat.v1.math.confusion_matrix(labels, predictions, num_classes=None, dtype=tf.int32, name=None, weights=None)</code></a>
* <a href="../../tf/compat/v1/count_nonzero.md"><code>tf.compat.v1.math.count_nonzero(input_tensor=None, axis=None, keepdims=None, dtype=tf.int64, name=None, reduction_indices=None, keep_dims=None, input=None)</code></a>
* <a href="../../tf/compat/v1/math/in_top_k.md"><code>tf.compat.v1.math.in_top_k(predictions, targets, k, name=None)</code></a>
* <a href="../../tf/compat/v1/reduce_all.md"><code>tf.compat.v1.math.reduce_all(input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None, keep_dims=None)</code></a>
* <a href="../../tf/compat/v1/reduce_any.md"><code>tf.compat.v1.math.reduce_any(input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None, keep_dims=None)</code></a>
* <a href="../../tf/compat/v1/reduce_logsumexp.md"><code>tf.compat.v1.math.reduce_logsumexp(input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None, keep_dims=None)</code></a>
* <a href="../../tf/compat/v1/reduce_max.md"><code>tf.compat.v1.math.reduce_max(input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None, keep_dims=None)</code></a>
* <a href="../../tf/compat/v1/reduce_mean.md"><code>tf.compat.v1.math.reduce_mean(input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None, keep_dims=None)</code></a>
* <a href="../../tf/compat/v1/reduce_min.md"><code>tf.compat.v1.math.reduce_min(input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None, keep_dims=None)</code></a>
* <a href="../../tf/compat/v1/reduce_prod.md"><code>tf.compat.v1.math.reduce_prod(input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None, keep_dims=None)</code></a>
* <a href="../../tf/compat/v1/reduce_sum.md"><code>tf.compat.v1.math.reduce_sum(input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None, keep_dims=None)</code></a>
* <a href="../../tf/compat/v1/scalar_mul.md"><code>tf.compat.v1.math.scalar_mul(scalar, x, name=None)</code></a>
* `tf.compat.v1.nn.avg_pool(value, ksize, strides, padding, data_format='NHWC', name=None, input=None)`
* <a href="../../tf/compat/v1/nn/batch_norm_with_global_normalization.md"><code>tf.compat.v1.nn.batch_norm_with_global_normalization(t=None, m=None, v=None, beta=None, gamma=None, variance_epsilon=None, scale_after_normalization=None, name=None, input=None, mean=None, variance=None)</code></a>
* <a href="../../tf/compat/v1/nn/bidirectional_dynamic_rnn.md"><code>tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None, initial_state_fw=None, initial_state_bw=None, dtype=None, parallel_iterations=None, swap_memory=False, time_major=False, scope=None)</code></a>
* <a href="../../tf/compat/v1/nn/conv1d.md"><code>tf.compat.v1.nn.conv1d(value=None, filters=None, stride=None, padding=None, use_cudnn_on_gpu=None, data_format=None, name=None, input=None, dilations=None)</code></a>
* `tf.compat.v1.nn.conv2d(input, filter=None, strides=None, padding=None, use_cudnn_on_gpu=True, data_format='NHWC', dilations=[1, 1, 1, 1], name=None, filters=None)`
* `tf.compat.v1.nn.conv2d_backprop_filter(input, filter_sizes, out_backprop, strides, padding, use_cudnn_on_gpu=True, data_format='NHWC', dilations=[1, 1, 1, 1], name=None)`
* `tf.compat.v1.nn.conv2d_backprop_input(input_sizes, filter=None, out_backprop=None, strides=None, padding=None, use_cudnn_on_gpu=True, data_format='NHWC', dilations=[1, 1, 1, 1], name=None, filters=None)`
* `tf.compat.v1.nn.conv2d_transpose(value=None, filter=None, output_shape=None, strides=None, padding='SAME', data_format='NHWC', name=None, input=None, filters=None, dilations=None)`
* `tf.compat.v1.nn.conv3d(input, filter=None, strides=None, padding=None, data_format='NDHWC', dilations=[1, 1, 1, 1, 1], name=None, filters=None)`
* `tf.compat.v1.nn.conv3d_backprop_filter(input, filter_sizes, out_backprop, strides, padding, data_format='NDHWC', dilations=[1, 1, 1, 1, 1], name=None)`
* `tf.compat.v1.nn.conv3d_transpose(value, filter=None, output_shape=None, strides=None, padding='SAME', data_format='NDHWC', name=None, input=None, filters=None, dilations=None)`
* <a href="../../tf/compat/v1/nn/convolution.md"><code>tf.compat.v1.nn.convolution(input, filter, padding, strides=None, dilation_rate=None, name=None, data_format=None, filters=None, dilations=None)</code></a>
* `tf.compat.v1.nn.crelu(features, name=None, axis=-1)`
* <a href="../../tf/compat/v1/nn/ctc_beam_search_decoder.md"><code>tf.compat.v1.nn.ctc_beam_search_decoder(inputs, sequence_length, beam_width=100, top_paths=1, merge_repeated=True)</code></a>
* <a href="../../tf/compat/v1/nn/ctc_loss.md"><code>tf.compat.v1.nn.ctc_loss(labels, inputs=None, sequence_length=None, preprocess_collapse_repeated=False, ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=False, time_major=True, logits=None)</code></a>
* <a href="../../tf/compat/v1/nn/ctc_loss_v2.md"><code>tf.compat.v1.nn.ctc_loss_v2(labels, logits, label_length, logit_length, logits_time_major=True, unique=None, blank_index=None, name=None)</code></a>
* `tf.compat.v1.nn.depth_to_space(input, block_size, name=None, data_format='NHWC')`
* <a href="../../tf/compat/v1/nn/depthwise_conv2d.md"><code>tf.compat.v1.nn.depthwise_conv2d(input, filter, strides, padding, rate=None, name=None, data_format=None, dilations=None)</code></a>
* `tf.compat.v1.nn.depthwise_conv2d_native(input, filter, strides, padding, data_format='NHWC', dilations=[1, 1, 1, 1], name=None)`
* <a href="../../tf/compat/v1/nn/dilation2d.md"><code>tf.compat.v1.nn.dilation2d(input, filter=None, strides=None, rates=None, padding=None, name=None, filters=None, dilations=None)</code></a>
* <a href="../../tf/compat/v1/nn/dropout.md"><code>tf.compat.v1.nn.dropout(x, keep_prob=None, noise_shape=None, seed=None, name=None, rate=None)</code></a>
* <a href="../../tf/compat/v1/nn/dynamic_rnn.md"><code>tf.compat.v1.nn.dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None, dtype=None, parallel_iterations=None, swap_memory=False, time_major=False, scope=None)</code></a>
* `tf.compat.v1.nn.embedding_lookup(params, ids, partition_strategy='mod', name=None, validate_indices=True, max_norm=None)`
* `tf.compat.v1.nn.embedding_lookup_sparse(params, sp_ids, sp_weights, partition_strategy='mod', name=None, combiner=None, max_norm=None)`
* <a href="../../tf/compat/v1/nn/erosion2d.md"><code>tf.compat.v1.nn.erosion2d(value, kernel, strides, rates, padding, name=None)</code></a>
* <a href="../../tf/compat/v1/nn/fractional_avg_pool.md"><code>tf.compat.v1.nn.fractional_avg_pool(value, pooling_ratio, pseudo_random=False, overlapping=False, deterministic=False, seed=0, seed2=0, name=None)</code></a>
* <a href="../../tf/compat/v1/nn/fractional_max_pool.md"><code>tf.compat.v1.nn.fractional_max_pool(value, pooling_ratio, pseudo_random=False, overlapping=False, deterministic=False, seed=0, seed2=0, name=None)</code></a>
* `tf.compat.v1.nn.fused_batch_norm(x, scale, offset, mean=None, variance=None, epsilon=0.001, data_format='NHWC', is_training=True, name=None, exponential_avg_factor=1.0)`
* <a href="../../tf/compat/v1/math/log_softmax.md"><code>tf.compat.v1.nn.log_softmax(logits, axis=None, name=None, dim=None)</code></a>
* `tf.compat.v1.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None, input=None)`
* `tf.compat.v1.nn.max_pool_with_argmax(input, ksize, strides, padding, data_format='NHWC', Targmax=None, name=None, output_dtype=None, include_batch_in_index=False)`
* <a href="../../tf/compat/v1/nn/moments.md"><code>tf.compat.v1.nn.moments(x, axes, shift=None, name=None, keep_dims=None, keepdims=None)</code></a>
* `tf.compat.v1.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, num_true=1, sampled_values=None, remove_accidental_hits=False, partition_strategy='mod', name='nce_loss')`
* <a href="../../tf/compat/v1/nn/pool.md"><code>tf.compat.v1.nn.pool(input, window_shape, pooling_type, padding, dilation_rate=None, strides=None, name=None, data_format=None, dilations=None)</code></a>
* <a href="../../tf/compat/v1/nn/quantized_avg_pool.md"><code>tf.compat.v1.nn.quantized_avg_pool(input, min_input, max_input, ksize, strides, padding, name=None)</code></a>
* `tf.compat.v1.nn.quantized_conv2d(input, filter, min_input, max_input, min_filter, max_filter, strides, padding, out_type=tf.qint32, dilations=[1, 1, 1, 1], name=None)`
* <a href="../../tf/compat/v1/nn/quantized_max_pool.md"><code>tf.compat.v1.nn.quantized_max_pool(input, min_input, max_input, ksize, strides, padding, name=None)</code></a>
* <a href="../../tf/compat/v1/nn/quantized_relu_x.md"><code>tf.compat.v1.nn.quantized_relu_x(features, max_value, min_features, max_features, out_type=tf.quint8, name=None)</code></a>
* <a href="../../tf/compat/v1/nn/raw_rnn.md"><code>tf.compat.v1.nn.raw_rnn(cell, loop_fn, parallel_iterations=None, swap_memory=False, scope=None)</code></a>
* <a href="../../tf/compat/v1/nn/relu_layer.md"><code>tf.compat.v1.nn.relu_layer(x, weights, biases, name=None)</code></a>
* `tf.compat.v1.nn.safe_embedding_lookup_sparse(embedding_weights, sparse_ids, sparse_weights=None, combiner='mean', default_id=None, name=None, partition_strategy='div', max_norm=None)`
* `tf.compat.v1.nn.sampled_softmax_loss(weights, biases, labels, inputs, num_sampled, num_classes, num_true=1, sampled_values=None, remove_accidental_hits=True, partition_strategy='mod', name='sampled_softmax_loss', seed=None)`
* <a href="../../tf/compat/v1/nn/separable_conv2d.md"><code>tf.compat.v1.nn.separable_conv2d(input, depthwise_filter, pointwise_filter, strides, padding, rate=None, name=None, data_format=None, dilations=None)</code></a>
* <a href="../../tf/compat/v1/nn/sigmoid_cross_entropy_with_logits.md"><code>tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)</code></a>
* <a href="../../tf/compat/v1/math/softmax.md"><code>tf.compat.v1.nn.softmax(logits, axis=None, name=None, dim=None)</code></a>
* `tf.compat.v1.nn.softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, dim=-1, name=None, axis=None)`
* <a href="../../tf/compat/v1/nn/softmax_cross_entropy_with_logits_v2.md"><code>tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels, logits, axis=None, name=None, dim=None)</code></a>
* <a href="../../tf/compat/v1/space_to_batch.md"><code>tf.compat.v1.nn.space_to_batch(input, paddings, block_size=None, name=None, block_shape=None)</code></a>
* `tf.compat.v1.nn.space_to_depth(input, block_size, name=None, data_format='NHWC')`
* <a href="../../tf/compat/v1/nn/sparse_softmax_cross_entropy_with_logits.md"><code>tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)</code></a>
* <a href="../../tf/compat/v1/nn/static_bidirectional_rnn.md"><code>tf.compat.v1.nn.static_bidirectional_rnn(cell_fw, cell_bw, inputs, initial_state_fw=None, initial_state_bw=None, dtype=None, sequence_length=None, scope=None)</code></a>
* <a href="../../tf/compat/v1/nn/static_rnn.md"><code>tf.compat.v1.nn.static_rnn(cell, inputs, initial_state=None, dtype=None, sequence_length=None, scope=None)</code></a>
* <a href="../../tf/compat/v1/nn/static_state_saving_rnn.md"><code>tf.compat.v1.nn.static_state_saving_rnn(cell, inputs, state_saver, state_name, sequence_length=None, scope=None)</code></a>
* <a href="../../tf/compat/v1/nn/sufficient_statistics.md"><code>tf.compat.v1.nn.sufficient_statistics(x, axes, shift=None, keep_dims=None, name=None, keepdims=None)</code></a>
* <a href="../../tf/compat/v1/nn/weighted_cross_entropy_with_logits.md"><code>tf.compat.v1.nn.weighted_cross_entropy_with_logits(labels=None, logits=None, pos_weight=None, name=None, targets=None)</code></a>
* <a href="../../tf/compat/v1/nn/weighted_moments.md"><code>tf.compat.v1.nn.weighted_moments(x, axes, frequency_weights, name=None, keep_dims=None, keepdims=None)</code></a>
* <a href="../../tf/compat/v1/nn/xw_plus_b.md"><code>tf.compat.v1.nn.xw_plus_b(x, weights, biases, name=None)</code></a>
* `tf.compat.v1.norm(tensor, ord='euclidean', axis=None, keepdims=None, name=None, keep_dims=None)`
* <a href="../../tf/compat/v1/ones_like.md"><code>tf.compat.v1.ones_like(tensor, dtype=None, name=None, optimize=True)</code></a>
* `tf.compat.v1.pad(tensor, paddings, mode='CONSTANT', name=None, constant_values=0)`
* <a href="../../tf/compat/v1/py_func.md"><code>tf.compat.v1.py_func(func, inp, Tout, stateful=True, name=None)</code></a>
* `tf.compat.v1.quantize_v2(input, min_range, max_range, T, mode='MIN_COMBINED', name=None, round_mode='HALF_AWAY_FROM_ZERO', narrow_range=False, axis=None, ensure_minimum_range=0.01)`
* `tf.compat.v1.ragged.constant_value(pylist, dtype=None, ragged_rank=None, inner_shape=None, row_splits_dtype='int64')`
* <a href="../../tf/compat/v1/ragged/placeholder.md"><code>tf.compat.v1.ragged.placeholder(dtype, ragged_rank, value_shape=None, name=None)</code></a>
* <a href="../../tf/compat/v1/multinomial.md"><code>tf.compat.v1.random.multinomial(logits, num_samples, seed=None, name=None, output_dtype=None)</code></a>
* <a href="../../tf/compat/v1/random_poisson.md"><code>tf.compat.v1.random.poisson(lam, shape, dtype=tf.float32, seed=None, name=None)</code></a>
* <a href="../../tf/compat/v1/random/stateless_multinomial.md"><code>tf.compat.v1.random.stateless_multinomial(logits, num_samples, seed, output_dtype=tf.int64, name=None)</code></a>
* <a href="../../tf/compat/v1/scan.md"><code>tf.compat.v1.scan(fn, elems, initializer=None, parallel_iterations=10, back_prop=True, swap_memory=False, infer_shape=True, reverse=False, name=None)</code></a>
* <a href="../../tf/compat/v1/setdiff1d.md"><code>tf.compat.v1.setdiff1d(x, y, index_dtype=tf.int32, name=None)</code></a>
* <a href="../../tf/compat/v1/shape.md"><code>tf.compat.v1.shape(input, name=None, out_type=tf.int32)</code></a>
* <a href="../../tf/compat/v1/size.md"><code>tf.compat.v1.size(input, name=None, out_type=tf.int32)</code></a>
* <a href="../../tf/compat/v1/sparse_to_dense.md"><code>tf.compat.v1.sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value=0, validate_indices=True, name=None)</code></a>
* <a href="../../tf/compat/v1/squeeze.md"><code>tf.compat.v1.squeeze(input, axis=None, name=None, squeeze_dims=None)</code></a>
* `tf.compat.v1.string_split(source, sep=None, skip_empty=True, delimiter=None, result_type='SparseTensor', name=None)`
* `tf.compat.v1.strings.length(input, name=None, unit='BYTE')`
* `tf.compat.v1.strings.reduce_join(inputs, axis=None, keep_dims=None, separator='', name=None, reduction_indices=None, keepdims=None)`
* `tf.compat.v1.strings.split(input=None, sep=None, maxsplit=-1, result_type='SparseTensor', source=None, name=None)`
* `tf.compat.v1.strings.substr(input, pos, len, name=None, unit='BYTE')`
* <a href="../../tf/compat/v1/string_to_hash_bucket.md"><code>tf.compat.v1.strings.to_hash_bucket(string_tensor=None, num_buckets=None, name=None, input=None)</code></a>
* <a href="../../tf/compat/v1/string_to_number.md"><code>tf.compat.v1.strings.to_number(string_tensor=None, out_type=tf.float32, name=None, input=None)</code></a>
* `tf.compat.v1.substr(input, pos, len, name=None, unit='BYTE')`
* `tf.compat.v1.to_bfloat16(x, name='ToBFloat16')`
* `tf.compat.v1.to_complex128(x, name='ToComplex128')`
* `tf.compat.v1.to_complex64(x, name='ToComplex64')`
* `tf.compat.v1.to_double(x, name='ToDouble')`
* `tf.compat.v1.to_float(x, name='ToFloat')`
* `tf.compat.v1.to_int32(x, name='ToInt32')`
* `tf.compat.v1.to_int64(x, name='ToInt64')`
* <a href="../../tf/compat/v1/train/sdca_fprint.md"><code>tf.compat.v1.train.sdca_fprint(input, name=None)</code></a>
* <a href="../../tf/compat/v1/train/sdca_optimizer.md"><code>tf.compat.v1.train.sdca_optimizer(sparse_example_indices, sparse_feature_indices, sparse_feature_values, dense_features, example_weights, example_labels, sparse_indices, sparse_weights, dense_weights, example_state_data, loss_type, l1, l2, num_loss_partitions, num_inner_iterations, adaptative=True, name=None)</code></a>
* <a href="../../tf/compat/v1/train/sdca_shrink_l1.md"><code>tf.compat.v1.train.sdca_shrink_l1(weights, l1, l2, name=None)</code></a>
* `tf.compat.v1.transpose(a, perm=None, name='transpose', conjugate=False)`
* <a href="../../tf/compat/v1/tuple.md"><code>tf.compat.v1.tuple(tensors, name=None, control_inputs=None)</code></a>
* <a href="../../tf/compat/v1/where.md"><code>tf.compat.v1.where(condition, x=None, y=None, name=None)</code></a>
* <a href="../../tf/compat/v1/zeros_like.md"><code>tf.compat.v1.zeros_like(tensor, dtype=None, name=None, optimize=True)</code></a>
* `tf.concat(values, axis, name='concat')`
* <a href="../../tf/cond.md"><code>tf.cond(pred, true_fn=None, false_fn=None, name=None)</code></a>
* <a href="../../tf/convert_to_tensor.md"><code>tf.convert_to_tensor(value, dtype=None, dtype_hint=None, name=None)</code></a>
* <a href="../../tf/debugging/Assert.md"><code>tf.debugging.Assert(condition, data, summarize=None, name=None)</code></a>
* <a href="../../tf/debugging/assert_all_finite.md"><code>tf.debugging.assert_all_finite(x, message, name=None)</code></a>
* <a href="../../tf/debugging/assert_equal.md"><code>tf.debugging.assert_equal(x, y, message=None, summarize=None, name=None)</code></a>
* <a href="../../tf/debugging/assert_greater.md"><code>tf.debugging.assert_greater(x, y, message=None, summarize=None, name=None)</code></a>
* <a href="../../tf/debugging/assert_greater_equal.md"><code>tf.debugging.assert_greater_equal(x, y, message=None, summarize=None, name=None)</code></a>
* <a href="../../tf/debugging/assert_integer.md"><code>tf.debugging.assert_integer(x, message=None, name=None)</code></a>
* <a href="../../tf/debugging/assert_less.md"><code>tf.debugging.assert_less(x, y, message=None, summarize=None, name=None)</code></a>
* <a href="../../tf/debugging/assert_less_equal.md"><code>tf.debugging.assert_less_equal(x, y, message=None, summarize=None, name=None)</code></a>
* <a href="../../tf/debugging/assert_near.md"><code>tf.debugging.assert_near(x, y, rtol=None, atol=None, message=None, summarize=None, name=None)</code></a>
* <a href="../../tf/debugging/assert_negative.md"><code>tf.debugging.assert_negative(x, message=None, summarize=None, name=None)</code></a>
* <a href="../../tf/debugging/assert_non_negative.md"><code>tf.debugging.assert_non_negative(x, message=None, summarize=None, name=None)</code></a>
* <a href="../../tf/debugging/assert_non_positive.md"><code>tf.debugging.assert_non_positive(x, message=None, summarize=None, name=None)</code></a>
* <a href="../../tf/debugging/assert_none_equal.md"><code>tf.debugging.assert_none_equal(x, y, summarize=None, message=None, name=None)</code></a>
* <a href="../../tf/debugging/assert_positive.md"><code>tf.debugging.assert_positive(x, message=None, summarize=None, name=None)</code></a>
* <a href="../../tf/debugging/assert_proper_iterable.md"><code>tf.debugging.assert_proper_iterable(values)</code></a>
* <a href="../../tf/debugging/assert_rank.md"><code>tf.debugging.assert_rank(x, rank, message=None, name=None)</code></a>
* <a href="../../tf/debugging/assert_rank_at_least.md"><code>tf.debugging.assert_rank_at_least(x, rank, message=None, name=None)</code></a>
* <a href="../../tf/debugging/assert_rank_in.md"><code>tf.debugging.assert_rank_in(x, ranks, message=None, name=None)</code></a>
* <a href="../../tf/debugging/assert_same_float_dtype.md"><code>tf.debugging.assert_same_float_dtype(tensors=None, dtype=None)</code></a>
* <a href="../../tf/debugging/assert_scalar.md"><code>tf.debugging.assert_scalar(tensor, message=None, name=None)</code></a>
* <a href="../../tf/debugging/assert_shapes.md"><code>tf.debugging.assert_shapes(shapes, data=None, summarize=None, message=None, name=None)</code></a>
* <a href="../../tf/debugging/assert_type.md"><code>tf.debugging.assert_type(tensor, tf_type, message=None, name=None)</code></a>
* <a href="../../tf/debugging/check_numerics.md"><code>tf.debugging.check_numerics(tensor, message, name=None)</code></a>
* <a href="../../tf/dtypes/complex.md"><code>tf.dtypes.complex(real, imag, name=None)</code></a>
* <a href="../../tf/dtypes/saturate_cast.md"><code>tf.dtypes.saturate_cast(value, dtype, name=None)</code></a>
* <a href="../../tf/dynamic_partition.md"><code>tf.dynamic_partition(data, partitions, num_partitions, name=None)</code></a>
* <a href="../../tf/dynamic_stitch.md"><code>tf.dynamic_stitch(indices, data, name=None)</code></a>
* `tf.edit_distance(hypothesis, truth, normalize=True, name='edit_distance')`
* <a href="../../tf/ensure_shape.md"><code>tf.ensure_shape(x, shape, name=None)</code></a>
* <a href="../../tf/expand_dims.md"><code>tf.expand_dims(input, axis, name=None)</code></a>
* <a href="../../tf/extract_volume_patches.md"><code>tf.extract_volume_patches(input, ksizes, strides, padding, name=None)</code></a>
* <a href="../../tf/eye.md"><code>tf.eye(num_rows, num_columns=None, batch_shape=None, dtype=tf.float32, name=None)</code></a>
* <a href="../../tf/fill.md"><code>tf.fill(dims, value, name=None)</code></a>
* `tf.fingerprint(data, method='farmhash64', name=None)`
* <a href="../../tf/foldl.md"><code>tf.foldl(fn, elems, initializer=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)</code></a>
* <a href="../../tf/foldr.md"><code>tf.foldr(fn, elems, initializer=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)</code></a>
* <a href="../../tf/gather.md"><code>tf.gather(params, indices, validate_indices=None, axis=None, batch_dims=0, name=None)</code></a>
* <a href="../../tf/gather_nd.md"><code>tf.gather_nd(params, indices, batch_dims=0, name=None)</code></a>
* <a href="../../tf/histogram_fixed_width.md"><code>tf.histogram_fixed_width(values, value_range, nbins=100, dtype=tf.int32, name=None)</code></a>
* <a href="../../tf/histogram_fixed_width_bins.md"><code>tf.histogram_fixed_width_bins(values, value_range, nbins=100, dtype=tf.int32, name=None)</code></a>
* <a href="../../tf/identity.md"><code>tf.identity(input, name=None)</code></a>
* <a href="../../tf/identity_n.md"><code>tf.identity_n(input, name=None)</code></a>
* <a href="../../tf/image/adjust_brightness.md"><code>tf.image.adjust_brightness(image, delta)</code></a>
* <a href="../../tf/image/adjust_contrast.md"><code>tf.image.adjust_contrast(images, contrast_factor)</code></a>
* <a href="../../tf/image/adjust_gamma.md"><code>tf.image.adjust_gamma(image, gamma=1, gain=1)</code></a>
* <a href="../../tf/image/adjust_hue.md"><code>tf.image.adjust_hue(image, delta, name=None)</code></a>
* <a href="../../tf/image/adjust_jpeg_quality.md"><code>tf.image.adjust_jpeg_quality(image, jpeg_quality, name=None)</code></a>
* <a href="../../tf/image/adjust_saturation.md"><code>tf.image.adjust_saturation(image, saturation_factor, name=None)</code></a>
* <a href="../../tf/image/central_crop.md"><code>tf.image.central_crop(image, central_fraction)</code></a>
* `tf.image.combined_non_max_suppression(boxes, scores, max_output_size_per_class, max_total_size, iou_threshold=0.5, score_threshold=-inf, pad_per_class=False, clip_boxes=True, name=None)`
* <a href="../../tf/image/convert_image_dtype.md"><code>tf.image.convert_image_dtype(image, dtype, saturate=False, name=None)</code></a>
* `tf.image.crop_and_resize(image, boxes, box_indices, crop_size, method='bilinear', extrapolation_value=0.0, name=None)`
* <a href="../../tf/image/crop_to_bounding_box.md"><code>tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)</code></a>
* <a href="../../tf/image/draw_bounding_boxes.md"><code>tf.image.draw_bounding_boxes(images, boxes, colors, name=None)</code></a>
* `tf.image.extract_glimpse(input, size, offsets, centered=True, normalized=True, noise='uniform', name=None)`
* <a href="../../tf/image/extract_patches.md"><code>tf.image.extract_patches(images, sizes, strides, rates, padding, name=None)</code></a>
* <a href="../../tf/image/flip_left_right.md"><code>tf.image.flip_left_right(image)</code></a>
* <a href="../../tf/image/flip_up_down.md"><code>tf.image.flip_up_down(image)</code></a>
* <a href="../../tf/image/generate_bounding_box_proposals.md"><code>tf.image.generate_bounding_box_proposals(scores, bbox_deltas, image_info, anchors, nms_threshold=0.7, pre_nms_topn=6000, min_size=16, post_nms_topn=300, name=None)</code></a>
* <a href="../../tf/image/grayscale_to_rgb.md"><code>tf.image.grayscale_to_rgb(images, name=None)</code></a>
* <a href="../../tf/image/hsv_to_rgb.md"><code>tf.image.hsv_to_rgb(images, name=None)</code></a>
* <a href="../../tf/image/image_gradients.md"><code>tf.image.image_gradients(image)</code></a>
* `tf.image.non_max_suppression(boxes, scores, max_output_size, iou_threshold=0.5, score_threshold=-inf, name=None)`
* `tf.image.non_max_suppression_overlaps(overlaps, scores, max_output_size, overlap_threshold=0.5, score_threshold=-inf, name=None)`
* `tf.image.non_max_suppression_padded(boxes, scores, max_output_size, iou_threshold=0.5, score_threshold=-inf, pad_to_max_output_size=False, name=None, sorted_input=False, canonicalized_coordinates=False, tile_size=512)`
* `tf.image.non_max_suppression_with_scores(boxes, scores, max_output_size, iou_threshold=0.5, score_threshold=-inf, soft_nms_sigma=0.0, name=None)`
* <a href="../../tf/image/pad_to_bounding_box.md"><code>tf.image.pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width)</code></a>
* <a href="../../tf/image/per_image_standardization.md"><code>tf.image.per_image_standardization(image)</code></a>
* <a href="../../tf/image/psnr.md"><code>tf.image.psnr(a, b, max_val, name=None)</code></a>
* <a href="../../tf/image/random_brightness.md"><code>tf.image.random_brightness(image, max_delta, seed=None)</code></a>
* <a href="../../tf/image/random_contrast.md"><code>tf.image.random_contrast(image, lower, upper, seed=None)</code></a>
* <a href="../../tf/image/random_crop.md"><code>tf.image.random_crop(value, size, seed=None, name=None)</code></a>
* <a href="../../tf/image/random_flip_left_right.md"><code>tf.image.random_flip_left_right(image, seed=None)</code></a>
* <a href="../../tf/image/random_flip_up_down.md"><code>tf.image.random_flip_up_down(image, seed=None)</code></a>
* <a href="../../tf/image/random_hue.md"><code>tf.image.random_hue(image, max_delta, seed=None)</code></a>
* <a href="../../tf/image/random_jpeg_quality.md"><code>tf.image.random_jpeg_quality(image, min_jpeg_quality, max_jpeg_quality, seed=None)</code></a>
* <a href="../../tf/image/random_saturation.md"><code>tf.image.random_saturation(image, lower, upper, seed=None)</code></a>
* `tf.image.resize(images, size, method='bilinear', preserve_aspect_ratio=False, antialias=False, name=None)`
* <a href="../../tf/image/resize_with_crop_or_pad.md"><code>tf.image.resize_with_crop_or_pad(image, target_height, target_width)</code></a>
* `tf.image.resize_with_pad(image, target_height, target_width, method='bilinear', antialias=False)`
* <a href="../../tf/image/rgb_to_grayscale.md"><code>tf.image.rgb_to_grayscale(images, name=None)</code></a>
* <a href="../../tf/image/rgb_to_hsv.md"><code>tf.image.rgb_to_hsv(images, name=None)</code></a>
* <a href="../../tf/image/rgb_to_yiq.md"><code>tf.image.rgb_to_yiq(images)</code></a>
* <a href="../../tf/image/rgb_to_yuv.md"><code>tf.image.rgb_to_yuv(images)</code></a>
* <a href="../../tf/image/rot90.md"><code>tf.image.rot90(image, k=1, name=None)</code></a>
* <a href="../../tf/image/sample_distorted_bounding_box.md"><code>tf.image.sample_distorted_bounding_box(image_size, bounding_boxes, seed=0, min_object_covered=0.1, aspect_ratio_range=None, area_range=None, max_attempts=None, use_image_if_no_bounding_boxes=None, name=None)</code></a>
* <a href="../../tf/image/sobel_edges.md"><code>tf.image.sobel_edges(image)</code></a>
* <a href="../../tf/image/ssim.md"><code>tf.image.ssim(img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)</code></a>
* `tf.image.ssim_multiscale(img1, img2, max_val, power_factors=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333), filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)`
* <a href="../../tf/image/stateless_random_brightness.md"><code>tf.image.stateless_random_brightness(image, max_delta, seed)</code></a>
* <a href="../../tf/image/stateless_random_contrast.md"><code>tf.image.stateless_random_contrast(image, lower, upper, seed)</code></a>
* <a href="../../tf/image/stateless_random_crop.md"><code>tf.image.stateless_random_crop(value, size, seed, name=None)</code></a>
* <a href="../../tf/image/stateless_random_flip_left_right.md"><code>tf.image.stateless_random_flip_left_right(image, seed)</code></a>
* <a href="../../tf/image/stateless_random_flip_up_down.md"><code>tf.image.stateless_random_flip_up_down(image, seed)</code></a>
* <a href="../../tf/image/stateless_random_hue.md"><code>tf.image.stateless_random_hue(image, max_delta, seed)</code></a>
* <a href="../../tf/image/stateless_random_jpeg_quality.md"><code>tf.image.stateless_random_jpeg_quality(image, min_jpeg_quality, max_jpeg_quality, seed)</code></a>
* <a href="../../tf/image/stateless_random_saturation.md"><code>tf.image.stateless_random_saturation(image, lower, upper, seed=None)</code></a>
* <a href="../../tf/image/stateless_sample_distorted_bounding_box.md"><code>tf.image.stateless_sample_distorted_bounding_box(image_size, bounding_boxes, seed, min_object_covered=0.1, aspect_ratio_range=None, area_range=None, max_attempts=None, use_image_if_no_bounding_boxes=None, name=None)</code></a>
* <a href="../../tf/image/total_variation.md"><code>tf.image.total_variation(images, name=None)</code></a>
* <a href="../../tf/image/transpose.md"><code>tf.image.transpose(image, name=None)</code></a>
* <a href="../../tf/image/yiq_to_rgb.md"><code>tf.image.yiq_to_rgb(images)</code></a>
* <a href="../../tf/image/yuv_to_rgb.md"><code>tf.image.yuv_to_rgb(images)</code></a>
* `tf.io.decode_and_crop_jpeg(contents, crop_window, channels=0, ratio=1, fancy_upscaling=True, try_recover_truncated=False, acceptable_fraction=1, dct_method='', name=None)`
* <a href="../../tf/io/decode_base64.md"><code>tf.io.decode_base64(input, name=None)</code></a>
* <a href="../../tf/io/decode_bmp.md"><code>tf.io.decode_bmp(contents, channels=0, name=None)</code></a>
* `tf.io.decode_compressed(bytes, compression_type='', name=None)`
* `tf.io.decode_csv(records, record_defaults, field_delim=',', use_quote_delim=True, na_value='', select_cols=None, name=None)`
* <a href="../../tf/io/decode_gif.md"><code>tf.io.decode_gif(contents, name=None)</code></a>
* <a href="../../tf/io/decode_image.md"><code>tf.io.decode_image(contents, channels=None, dtype=tf.uint8, name=None, expand_animations=True)</code></a>
* `tf.io.decode_jpeg(contents, channels=0, ratio=1, fancy_upscaling=True, try_recover_truncated=False, acceptable_fraction=1, dct_method='', name=None)`
* <a href="../../tf/io/decode_png.md"><code>tf.io.decode_png(contents, channels=0, dtype=tf.uint8, name=None)</code></a>
* `tf.io.decode_proto(bytes, message_type, field_names, output_types, descriptor_source='local://', message_format='binary', sanitize=False, name=None)`
* <a href="../../tf/io/decode_raw.md"><code>tf.io.decode_raw(input_bytes, out_type, little_endian=True, fixed_length=None, name=None)</code></a>
* <a href="../../tf/io/deserialize_many_sparse.md"><code>tf.io.deserialize_many_sparse(serialized_sparse, dtype, rank=None, name=None)</code></a>
* <a href="../../tf/io/encode_base64.md"><code>tf.io.encode_base64(input, pad=False, name=None)</code></a>
* `tf.io.encode_jpeg(image, format='', quality=95, progressive=False, optimize_size=False, chroma_downsampling=True, density_unit='in', x_density=300, y_density=300, xmp_metadata='', name=None)`
* `tf.io.encode_png(image, compression=-1, name=None)`
* `tf.io.encode_proto(sizes, values, field_names, message_type, descriptor_source='local://', name=None)`
* <a href="../../tf/io/extract_jpeg_shape.md"><code>tf.io.extract_jpeg_shape(contents, output_type=tf.int32, name=None)</code></a>
* <a href="../../tf/io/matching_files.md"><code>tf.io.matching_files(pattern, name=None)</code></a>
* <a href="../../tf/io/parse_example.md"><code>tf.io.parse_example(serialized, features, example_names=None, name=None)</code></a>
* <a href="../../tf/io/parse_sequence_example.md"><code>tf.io.parse_sequence_example(serialized, context_features=None, sequence_features=None, example_names=None, name=None)</code></a>
* <a href="../../tf/io/parse_single_example.md"><code>tf.io.parse_single_example(serialized, features, example_names=None, name=None)</code></a>
* <a href="../../tf/io/parse_single_sequence_example.md"><code>tf.io.parse_single_sequence_example(serialized, context_features=None, sequence_features=None, example_name=None, name=None)</code></a>
* <a href="../../tf/io/parse_tensor.md"><code>tf.io.parse_tensor(serialized, out_type, name=None)</code></a>
* <a href="../../tf/io/serialize_many_sparse.md"><code>tf.io.serialize_many_sparse(sp_input, out_type=tf.string, name=None)</code></a>
* <a href="../../tf/io/serialize_sparse.md"><code>tf.io.serialize_sparse(sp_input, out_type=tf.string, name=None)</code></a>
* <a href="../../tf/io/write_file.md"><code>tf.io.write_file(filename, contents, name=None)</code></a>
* <a href="../../tf/linalg/adjoint.md"><code>tf.linalg.adjoint(matrix, name=None)</code></a>
* <a href="../../tf/linalg/band_part.md"><code>tf.linalg.band_part(input, num_lower, num_upper, name=None)</code></a>
* <a href="../../tf/linalg/cholesky.md"><code>tf.linalg.cholesky(input, name=None)</code></a>
* <a href="../../tf/linalg/cholesky_solve.md"><code>tf.linalg.cholesky_solve(chol, rhs, name=None)</code></a>
* <a href="../../tf/linalg/cross.md"><code>tf.linalg.cross(a, b, name=None)</code></a>
* <a href="../../tf/linalg/det.md"><code>tf.linalg.det(input, name=None)</code></a>
* `tf.linalg.diag(diagonal, name='diag', k=0, num_rows=-1, num_cols=-1, padding_value=0, align='RIGHT_LEFT')`
* `tf.linalg.diag_part(input, name='diag_part', k=0, padding_value=0, align='RIGHT_LEFT')`
* <a href="../../tf/linalg/eig.md"><code>tf.linalg.eig(tensor, name=None)</code></a>
* <a href="../../tf/linalg/eigh.md"><code>tf.linalg.eigh(tensor, name=None)</code></a>
* `tf.linalg.eigh_tridiagonal(alpha, beta, eigvals_only=True, select='a', select_range=None, tol=None, name=None)`
* <a href="../../tf/linalg/eigvals.md"><code>tf.linalg.eigvals(tensor, name=None)</code></a>
* <a href="../../tf/linalg/eigvalsh.md"><code>tf.linalg.eigvalsh(tensor, name=None)</code></a>
* `tf.linalg.experimental.conjugate_gradient(operator, rhs, preconditioner=None, x=None, tol=1e-05, max_iter=20, name='conjugate_gradient')`
* <a href="../../tf/linalg/expm.md"><code>tf.linalg.expm(input, name=None)</code></a>
* <a href="../../tf/linalg/global_norm.md"><code>tf.linalg.global_norm(t_list, name=None)</code></a>
* <a href="../../tf/linalg/inv.md"><code>tf.linalg.inv(input, adjoint=False, name=None)</code></a>
* <a href="../../tf/linalg/logdet.md"><code>tf.linalg.logdet(matrix, name=None)</code></a>
* <a href="../../tf/linalg/logm.md"><code>tf.linalg.logm(input, name=None)</code></a>
* <a href="../../tf/linalg/lstsq.md"><code>tf.linalg.lstsq(matrix, rhs, l2_regularizer=0.0, fast=True, name=None)</code></a>
* <a href="../../tf/linalg/lu.md"><code>tf.linalg.lu(input, output_idx_type=tf.int32, name=None)</code></a>
* <a href="../../tf/linalg/lu_matrix_inverse.md"><code>tf.linalg.lu_matrix_inverse(lower_upper, perm, validate_args=False, name=None)</code></a>
* <a href="../../tf/linalg/lu_reconstruct.md"><code>tf.linalg.lu_reconstruct(lower_upper, perm, validate_args=False, name=None)</code></a>
* <a href="../../tf/linalg/lu_solve.md"><code>tf.linalg.lu_solve(lower_upper, perm, rhs, validate_args=False, name=None)</code></a>
* <a href="../../tf/linalg/matmul.md"><code>tf.linalg.matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, output_type=None, name=None)</code></a>
* <a href="../../tf/linalg/matrix_rank.md"><code>tf.linalg.matrix_rank(a, tol=None, validate_args=False, name=None)</code></a>
* `tf.linalg.matrix_transpose(a, name='matrix_transpose', conjugate=False)`
* <a href="../../tf/linalg/matvec.md"><code>tf.linalg.matvec(a, b, transpose_a=False, adjoint_a=False, a_is_sparse=False, b_is_sparse=False, name=None)</code></a>
* `tf.linalg.normalize(tensor, ord='euclidean', axis=None, name=None)`
* <a href="../../tf/linalg/pinv.md"><code>tf.linalg.pinv(a, rcond=None, validate_args=False, name=None)</code></a>
* <a href="../../tf/linalg/qr.md"><code>tf.linalg.qr(input, full_matrices=False, name=None)</code></a>
* `tf.linalg.set_diag(input, diagonal, name='set_diag', k=0, align='RIGHT_LEFT')`
* <a href="../../tf/linalg/slogdet.md"><code>tf.linalg.slogdet(input, name=None)</code></a>
* <a href="../../tf/linalg/solve.md"><code>tf.linalg.solve(matrix, rhs, adjoint=False, name=None)</code></a>
* <a href="../../tf/linalg/sqrtm.md"><code>tf.linalg.sqrtm(input, name=None)</code></a>
* <a href="../../tf/linalg/svd.md"><code>tf.linalg.svd(tensor, full_matrices=False, compute_uv=True, name=None)</code></a>
* <a href="../../tf/linalg/tensor_diag.md"><code>tf.linalg.tensor_diag(diagonal, name=None)</code></a>
* <a href="../../tf/linalg/tensor_diag_part.md"><code>tf.linalg.tensor_diag_part(input, name=None)</code></a>
* <a href="../../tf/linalg/trace.md"><code>tf.linalg.trace(x, name=None)</code></a>
* <a href="../../tf/linalg/triangular_solve.md"><code>tf.linalg.triangular_solve(matrix, rhs, lower=True, adjoint=False, name=None)</code></a>
* `tf.linalg.tridiagonal_matmul(diagonals, rhs, diagonals_format='compact', name=None)`
* `tf.linalg.tridiagonal_solve(diagonals, rhs, diagonals_format='compact', transpose_rhs=False, conjugate_rhs=False, name=None, partial_pivoting=True, perturb_singular=False)`
* <a href="../../tf/linspace.md"><code>tf.linspace(start, stop, num, name=None, axis=0)</code></a>
* <a href="../../tf/math/abs.md"><code>tf.math.abs(x, name=None)</code></a>
* <a href="../../tf/math/accumulate_n.md"><code>tf.math.accumulate_n(inputs, shape=None, tensor_dtype=None, name=None)</code></a>
* <a href="../../tf/math/acos.md"><code>tf.math.acos(x, name=None)</code></a>
* <a href="../../tf/math/acosh.md"><code>tf.math.acosh(x, name=None)</code></a>
* <a href="../../tf/math/add.md"><code>tf.math.add(x, y, name=None)</code></a>
* <a href="../../tf/math/add_n.md"><code>tf.math.add_n(inputs, name=None)</code></a>
* <a href="../../tf/math/angle.md"><code>tf.math.angle(input, name=None)</code></a>
* <a href="../../tf/math/argmax.md"><code>tf.math.argmax(input, axis=None, output_type=tf.int64, name=None)</code></a>
* <a href="../../tf/math/argmin.md"><code>tf.math.argmin(input, axis=None, output_type=tf.int64, name=None)</code></a>
* <a href="../../tf/math/asin.md"><code>tf.math.asin(x, name=None)</code></a>
* <a href="../../tf/math/asinh.md"><code>tf.math.asinh(x, name=None)</code></a>
* <a href="../../tf/math/atan.md"><code>tf.math.atan(x, name=None)</code></a>
* <a href="../../tf/math/atan2.md"><code>tf.math.atan2(y, x, name=None)</code></a>
* <a href="../../tf/math/atanh.md"><code>tf.math.atanh(x, name=None)</code></a>
* <a href="../../tf/math/bessel_i0.md"><code>tf.math.bessel_i0(x, name=None)</code></a>
* <a href="../../tf/math/bessel_i0e.md"><code>tf.math.bessel_i0e(x, name=None)</code></a>
* <a href="../../tf/math/bessel_i1.md"><code>tf.math.bessel_i1(x, name=None)</code></a>
* <a href="../../tf/math/bessel_i1e.md"><code>tf.math.bessel_i1e(x, name=None)</code></a>
* <a href="../../tf/math/betainc.md"><code>tf.math.betainc(a, b, x, name=None)</code></a>
* <a href="../../tf/math/ceil.md"><code>tf.math.ceil(x, name=None)</code></a>
* <a href="../../tf/math/confusion_matrix.md"><code>tf.math.confusion_matrix(labels, predictions, num_classes=None, weights=None, dtype=tf.int32, name=None)</code></a>
* <a href="../../tf/math/conj.md"><code>tf.math.conj(x, name=None)</code></a>
* <a href="../../tf/math/cos.md"><code>tf.math.cos(x, name=None)</code></a>
* <a href="../../tf/math/cosh.md"><code>tf.math.cosh(x, name=None)</code></a>
* <a href="../../tf/math/count_nonzero.md"><code>tf.math.count_nonzero(input, axis=None, keepdims=None, dtype=tf.int64, name=None)</code></a>
* <a href="../../tf/math/cumprod.md"><code>tf.math.cumprod(x, axis=0, exclusive=False, reverse=False, name=None)</code></a>
* <a href="../../tf/math/cumsum.md"><code>tf.math.cumsum(x, axis=0, exclusive=False, reverse=False, name=None)</code></a>
* <a href="../../tf/math/cumulative_logsumexp.md"><code>tf.math.cumulative_logsumexp(x, axis=0, exclusive=False, reverse=False, name=None)</code></a>
* <a href="../../tf/math/digamma.md"><code>tf.math.digamma(x, name=None)</code></a>
* <a href="../../tf/math/divide.md"><code>tf.math.divide(x, y, name=None)</code></a>
* <a href="../../tf/math/divide_no_nan.md"><code>tf.math.divide_no_nan(x, y, name=None)</code></a>
* <a href="../../tf/math/equal.md"><code>tf.math.equal(x, y, name=None)</code></a>
* <a href="../../tf/math/erf.md"><code>tf.math.erf(x, name=None)</code></a>
* <a href="../../tf/math/erfc.md"><code>tf.math.erfc(x, name=None)</code></a>
* <a href="../../tf/math/erfcinv.md"><code>tf.math.erfcinv(x, name=None)</code></a>
* <a href="../../tf/math/erfinv.md"><code>tf.math.erfinv(x, name=None)</code></a>
* <a href="../../tf/math/exp.md"><code>tf.math.exp(x, name=None)</code></a>
* <a href="../../tf/math/expm1.md"><code>tf.math.expm1(x, name=None)</code></a>
* <a href="../../tf/math/floor.md"><code>tf.math.floor(x, name=None)</code></a>
* <a href="../../tf/math/floordiv.md"><code>tf.math.floordiv(x, y, name=None)</code></a>
* <a href="../../tf/math/floormod.md"><code>tf.math.floormod(x, y, name=None)</code></a>
* <a href="../../tf/math/greater.md"><code>tf.math.greater(x, y, name=None)</code></a>
* <a href="../../tf/math/greater_equal.md"><code>tf.math.greater_equal(x, y, name=None)</code></a>
* <a href="../../tf/math/igamma.md"><code>tf.math.igamma(a, x, name=None)</code></a>
* <a href="../../tf/math/igammac.md"><code>tf.math.igammac(a, x, name=None)</code></a>
* <a href="../../tf/math/imag.md"><code>tf.math.imag(input, name=None)</code></a>
* <a href="../../tf/math/in_top_k.md"><code>tf.math.in_top_k(targets, predictions, k, name=None)</code></a>
* <a href="../../tf/math/invert_permutation.md"><code>tf.math.invert_permutation(x, name=None)</code></a>
* <a href="../../tf/math/is_finite.md"><code>tf.math.is_finite(x, name=None)</code></a>
* <a href="../../tf/math/is_inf.md"><code>tf.math.is_inf(x, name=None)</code></a>
* <a href="../../tf/math/is_nan.md"><code>tf.math.is_nan(x, name=None)</code></a>
* <a href="../../tf/math/is_non_decreasing.md"><code>tf.math.is_non_decreasing(x, name=None)</code></a>
* <a href="../../tf/math/is_strictly_increasing.md"><code>tf.math.is_strictly_increasing(x, name=None)</code></a>
* `tf.math.l2_normalize(x, axis=None, epsilon=1e-12, name=None, dim=None)`
* <a href="../../tf/math/lbeta.md"><code>tf.math.lbeta(x, name=None)</code></a>
* <a href="../../tf/math/less.md"><code>tf.math.less(x, y, name=None)</code></a>
* <a href="../../tf/math/less_equal.md"><code>tf.math.less_equal(x, y, name=None)</code></a>
* <a href="../../tf/math/lgamma.md"><code>tf.math.lgamma(x, name=None)</code></a>
* <a href="../../tf/math/log.md"><code>tf.math.log(x, name=None)</code></a>
* <a href="../../tf/math/log1p.md"><code>tf.math.log1p(x, name=None)</code></a>
* <a href="../../tf/math/log_sigmoid.md"><code>tf.math.log_sigmoid(x, name=None)</code></a>
* <a href="../../tf/math/logical_and.md"><code>tf.math.logical_and(x, y, name=None)</code></a>
* <a href="../../tf/math/logical_not.md"><code>tf.math.logical_not(x, name=None)</code></a>
* <a href="../../tf/math/logical_or.md"><code>tf.math.logical_or(x, y, name=None)</code></a>
* `tf.math.logical_xor(x, y, name='LogicalXor')`
* <a href="../../tf/math/maximum.md"><code>tf.math.maximum(x, y, name=None)</code></a>
* <a href="../../tf/math/minimum.md"><code>tf.math.minimum(x, y, name=None)</code></a>
* <a href="../../tf/math/multiply.md"><code>tf.math.multiply(x, y, name=None)</code></a>
* <a href="../../tf/math/multiply_no_nan.md"><code>tf.math.multiply_no_nan(x, y, name=None)</code></a>
* <a href="../../tf/math/ndtri.md"><code>tf.math.ndtri(x, name=None)</code></a>
* <a href="../../tf/math/negative.md"><code>tf.math.negative(x, name=None)</code></a>
* <a href="../../tf/math/nextafter.md"><code>tf.math.nextafter(x1, x2, name=None)</code></a>
* <a href="../../tf/math/not_equal.md"><code>tf.math.not_equal(x, y, name=None)</code></a>
* <a href="../../tf/math/polygamma.md"><code>tf.math.polygamma(a, x, name=None)</code></a>
* <a href="../../tf/math/polyval.md"><code>tf.math.polyval(coeffs, x, name=None)</code></a>
* <a href="../../tf/math/pow.md"><code>tf.math.pow(x, y, name=None)</code></a>
* <a href="../../tf/math/real.md"><code>tf.math.real(input, name=None)</code></a>
* <a href="../../tf/math/reciprocal.md"><code>tf.math.reciprocal(x, name=None)</code></a>
* <a href="../../tf/math/reciprocal_no_nan.md"><code>tf.math.reciprocal_no_nan(x, name=None)</code></a>
* <a href="../../tf/math/reduce_all.md"><code>tf.math.reduce_all(input_tensor, axis=None, keepdims=False, name=None)</code></a>
* <a href="../../tf/math/reduce_any.md"><code>tf.math.reduce_any(input_tensor, axis=None, keepdims=False, name=None)</code></a>
* <a href="../../tf/math/reduce_euclidean_norm.md"><code>tf.math.reduce_euclidean_norm(input_tensor, axis=None, keepdims=False, name=None)</code></a>
* <a href="../../tf/math/reduce_logsumexp.md"><code>tf.math.reduce_logsumexp(input_tensor, axis=None, keepdims=False, name=None)</code></a>
* <a href="../../tf/math/reduce_max.md"><code>tf.math.reduce_max(input_tensor, axis=None, keepdims=False, name=None)</code></a>
* <a href="../../tf/math/reduce_mean.md"><code>tf.math.reduce_mean(input_tensor, axis=None, keepdims=False, name=None)</code></a>
* <a href="../../tf/math/reduce_min.md"><code>tf.math.reduce_min(input_tensor, axis=None, keepdims=False, name=None)</code></a>
* <a href="../../tf/math/reduce_prod.md"><code>tf.math.reduce_prod(input_tensor, axis=None, keepdims=False, name=None)</code></a>
* <a href="../../tf/math/reduce_std.md"><code>tf.math.reduce_std(input_tensor, axis=None, keepdims=False, name=None)</code></a>
* <a href="../../tf/math/reduce_sum.md"><code>tf.math.reduce_sum(input_tensor, axis=None, keepdims=False, name=None)</code></a>
* <a href="../../tf/math/reduce_variance.md"><code>tf.math.reduce_variance(input_tensor, axis=None, keepdims=False, name=None)</code></a>
* <a href="../../tf/math/rint.md"><code>tf.math.rint(x, name=None)</code></a>
* <a href="../../tf/math/round.md"><code>tf.math.round(x, name=None)</code></a>
* <a href="../../tf/math/rsqrt.md"><code>tf.math.rsqrt(x, name=None)</code></a>
* <a href="../../tf/math/scalar_mul.md"><code>tf.math.scalar_mul(scalar, x, name=None)</code></a>
* <a href="../../tf/math/segment_max.md"><code>tf.math.segment_max(data, segment_ids, name=None)</code></a>
* <a href="../../tf/math/segment_mean.md"><code>tf.math.segment_mean(data, segment_ids, name=None)</code></a>
* <a href="../../tf/math/segment_min.md"><code>tf.math.segment_min(data, segment_ids, name=None)</code></a>
* <a href="../../tf/math/segment_prod.md"><code>tf.math.segment_prod(data, segment_ids, name=None)</code></a>
* <a href="../../tf/math/segment_sum.md"><code>tf.math.segment_sum(data, segment_ids, name=None)</code></a>
* <a href="../../tf/math/sigmoid.md"><code>tf.math.sigmoid(x, name=None)</code></a>
* <a href="../../tf/math/sign.md"><code>tf.math.sign(x, name=None)</code></a>
* <a href="../../tf/math/sin.md"><code>tf.math.sin(x, name=None)</code></a>
* <a href="../../tf/math/sinh.md"><code>tf.math.sinh(x, name=None)</code></a>
* <a href="../../tf/math/sobol_sample.md"><code>tf.math.sobol_sample(dim, num_results, skip=0, dtype=tf.float32, name=None)</code></a>
* <a href="../../tf/math/softplus.md"><code>tf.math.softplus(features, name=None)</code></a>
* <a href="../../tf/math/special/bessel_j0.md"><code>tf.math.special.bessel_j0(x, name=None)</code></a>
* <a href="../../tf/math/special/bessel_j1.md"><code>tf.math.special.bessel_j1(x, name=None)</code></a>
* <a href="../../tf/math/special/bessel_k0.md"><code>tf.math.special.bessel_k0(x, name=None)</code></a>
* <a href="../../tf/math/special/bessel_k0e.md"><code>tf.math.special.bessel_k0e(x, name=None)</code></a>
* <a href="../../tf/math/special/bessel_k1.md"><code>tf.math.special.bessel_k1(x, name=None)</code></a>
* <a href="../../tf/math/special/bessel_k1e.md"><code>tf.math.special.bessel_k1e(x, name=None)</code></a>
* <a href="../../tf/math/special/bessel_y0.md"><code>tf.math.special.bessel_y0(x, name=None)</code></a>
* <a href="../../tf/math/special/bessel_y1.md"><code>tf.math.special.bessel_y1(x, name=None)</code></a>
* <a href="../../tf/math/special/dawsn.md"><code>tf.math.special.dawsn(x, name=None)</code></a>
* <a href="../../tf/math/special/expint.md"><code>tf.math.special.expint(x, name=None)</code></a>
* <a href="../../tf/math/special/fresnel_cos.md"><code>tf.math.special.fresnel_cos(x, name=None)</code></a>
* <a href="../../tf/math/special/fresnel_sin.md"><code>tf.math.special.fresnel_sin(x, name=None)</code></a>
* <a href="../../tf/math/special/spence.md"><code>tf.math.special.spence(x, name=None)</code></a>
* <a href="../../tf/math/sqrt.md"><code>tf.math.sqrt(x, name=None)</code></a>
* <a href="../../tf/math/square.md"><code>tf.math.square(x, name=None)</code></a>
* <a href="../../tf/math/squared_difference.md"><code>tf.math.squared_difference(x, y, name=None)</code></a>
* <a href="../../tf/math/subtract.md"><code>tf.math.subtract(x, y, name=None)</code></a>
* <a href="../../tf/math/tan.md"><code>tf.math.tan(x, name=None)</code></a>
* <a href="../../tf/math/tanh.md"><code>tf.math.tanh(x, name=None)</code></a>
* <a href="../../tf/math/top_k.md"><code>tf.math.top_k(input, k=1, sorted=True, name=None)</code></a>
* <a href="../../tf/math/truediv.md"><code>tf.math.truediv(x, y, name=None)</code></a>
* <a href="../../tf/math/unsorted_segment_max.md"><code>tf.math.unsorted_segment_max(data, segment_ids, num_segments, name=None)</code></a>
* <a href="../../tf/math/unsorted_segment_mean.md"><code>tf.math.unsorted_segment_mean(data, segment_ids, num_segments, name=None)</code></a>
* <a href="../../tf/math/unsorted_segment_min.md"><code>tf.math.unsorted_segment_min(data, segment_ids, num_segments, name=None)</code></a>
* <a href="../../tf/math/unsorted_segment_prod.md"><code>tf.math.unsorted_segment_prod(data, segment_ids, num_segments, name=None)</code></a>
* <a href="../../tf/math/unsorted_segment_sqrt_n.md"><code>tf.math.unsorted_segment_sqrt_n(data, segment_ids, num_segments, name=None)</code></a>
* <a href="../../tf/math/unsorted_segment_sum.md"><code>tf.math.unsorted_segment_sum(data, segment_ids, num_segments, name=None)</code></a>
* <a href="../../tf/math/xdivy.md"><code>tf.math.xdivy(x, y, name=None)</code></a>
* <a href="../../tf/math/xlog1py.md"><code>tf.math.xlog1py(x, y, name=None)</code></a>
* <a href="../../tf/math/xlogy.md"><code>tf.math.xlogy(x, y, name=None)</code></a>
* <a href="../../tf/math/zero_fraction.md"><code>tf.math.zero_fraction(value, name=None)</code></a>
* <a href="../../tf/math/zeta.md"><code>tf.math.zeta(x, q, name=None)</code></a>
* <a href="../../tf/nn/atrous_conv2d.md"><code>tf.nn.atrous_conv2d(value, filters, rate, padding, name=None)</code></a>
* <a href="../../tf/nn/atrous_conv2d_transpose.md"><code>tf.nn.atrous_conv2d_transpose(value, filters, output_shape, rate, padding, name=None)</code></a>
* <a href="../../tf/nn/avg_pool.md"><code>tf.nn.avg_pool(input, ksize, strides, padding, data_format=None, name=None)</code></a>
* `tf.nn.avg_pool1d(input, ksize, strides, padding, data_format='NWC', name=None)`
* `tf.nn.avg_pool2d(input, ksize, strides, padding, data_format='NHWC', name=None)`
* `tf.nn.avg_pool3d(input, ksize, strides, padding, data_format='NDHWC', name=None)`
* <a href="../../tf/nn/batch_norm_with_global_normalization.md"><code>tf.nn.batch_norm_with_global_normalization(input, mean, variance, beta, gamma, variance_epsilon, scale_after_normalization, name=None)</code></a>
* <a href="../../tf/nn/batch_normalization.md"><code>tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None)</code></a>
* <a href="../../tf/nn/bias_add.md"><code>tf.nn.bias_add(value, bias, data_format=None, name=None)</code></a>
* <a href="../../tf/nn/collapse_repeated.md"><code>tf.nn.collapse_repeated(labels, seq_length, name=None)</code></a>
* <a href="../../tf/nn/compute_accidental_hits.md"><code>tf.nn.compute_accidental_hits(true_classes, sampled_candidates, num_true, seed=None, name=None)</code></a>
* <a href="../../tf/nn/compute_average_loss.md"><code>tf.nn.compute_average_loss(per_example_loss, sample_weight=None, global_batch_size=None)</code></a>
* `tf.nn.conv1d(input, filters, stride, padding, data_format='NWC', dilations=None, name=None)`
* `tf.nn.conv1d_transpose(input, filters, output_shape, strides, padding='SAME', data_format='NWC', dilations=None, name=None)`
* `tf.nn.conv2d(input, filters, strides, padding, data_format='NHWC', dilations=None, name=None)`
* `tf.nn.conv2d_transpose(input, filters, output_shape, strides, padding='SAME', data_format='NHWC', dilations=None, name=None)`
* `tf.nn.conv3d(input, filters, strides, padding, data_format='NDHWC', dilations=None, name=None)`
* `tf.nn.conv3d_transpose(input, filters, output_shape, strides, padding='SAME', data_format='NDHWC', dilations=None, name=None)`
* `tf.nn.conv_transpose(input, filters, output_shape, strides, padding='SAME', data_format=None, dilations=None, name=None)`
* `tf.nn.convolution(input, filters, strides=None, padding='VALID', data_format=None, dilations=None, name=None)`
* `tf.nn.crelu(features, axis=-1, name=None)`
* <a href="../../tf/nn/ctc_beam_search_decoder.md"><code>tf.nn.ctc_beam_search_decoder(inputs, sequence_length, beam_width=100, top_paths=1)</code></a>
* <a href="../../tf/nn/ctc_greedy_decoder.md"><code>tf.nn.ctc_greedy_decoder(inputs, sequence_length, merge_repeated=True, blank_index=None)</code></a>
* <a href="../../tf/nn/ctc_loss.md"><code>tf.nn.ctc_loss(labels, logits, label_length, logit_length, logits_time_major=True, unique=None, blank_index=None, name=None)</code></a>
* <a href="../../tf/nn/ctc_unique_labels.md"><code>tf.nn.ctc_unique_labels(labels, name=None)</code></a>
* `tf.nn.depth_to_space(input, block_size, data_format='NHWC', name=None)`
* <a href="../../tf/nn/depthwise_conv2d.md"><code>tf.nn.depthwise_conv2d(input, filter, strides, padding, data_format=None, dilations=None, name=None)</code></a>
* `tf.nn.depthwise_conv2d_backprop_filter(input, filter_sizes, out_backprop, strides, padding, data_format='NHWC', dilations=[1, 1, 1, 1], name=None)`
* `tf.nn.depthwise_conv2d_backprop_input(input_sizes, filter, out_backprop, strides, padding, data_format='NHWC', dilations=[1, 1, 1, 1], name=None)`
* <a href="../../tf/nn/dilation2d.md"><code>tf.nn.dilation2d(input, filters, strides, padding, data_format, dilations, name=None)</code></a>
* <a href="../../tf/nn/dropout.md"><code>tf.nn.dropout(x, rate, noise_shape=None, seed=None, name=None)</code></a>
* <a href="../../tf/nn/elu.md"><code>tf.nn.elu(features, name=None)</code></a>
* <a href="../../tf/nn/embedding_lookup.md"><code>tf.nn.embedding_lookup(params, ids, max_norm=None, name=None)</code></a>
* <a href="../../tf/nn/embedding_lookup_sparse.md"><code>tf.nn.embedding_lookup_sparse(params, sp_ids, sp_weights, combiner=None, max_norm=None, name=None)</code></a>
* <a href="../../tf/nn/erosion2d.md"><code>tf.nn.erosion2d(value, filters, strides, padding, data_format, dilations, name=None)</code></a>
* <a href="../../tf/nn/experimental/stateless_dropout.md"><code>tf.nn.experimental.stateless_dropout(x, rate, seed, rng_alg=None, noise_shape=None, name=None)</code></a>
* <a href="../../tf/nn/fractional_avg_pool.md"><code>tf.nn.fractional_avg_pool(value, pooling_ratio, pseudo_random=False, overlapping=False, seed=0, name=None)</code></a>
* <a href="../../tf/nn/fractional_max_pool.md"><code>tf.nn.fractional_max_pool(value, pooling_ratio, pseudo_random=False, overlapping=False, seed=0, name=None)</code></a>
* <a href="../../tf/nn/gelu.md"><code>tf.nn.gelu(features, approximate=False, name=None)</code></a>
* `tf.nn.isotonic_regression(inputs, decreasing=True, axis=-1)`
* <a href="../../tf/nn/l2_loss.md"><code>tf.nn.l2_loss(t, name=None)</code></a>
* <a href="../../tf/nn/leaky_relu.md"><code>tf.nn.leaky_relu(features, alpha=0.2, name=None)</code></a>
* <a href="../../tf/nn/local_response_normalization.md"><code>tf.nn.local_response_normalization(input, depth_radius=5, bias=1, alpha=1, beta=0.5, name=None)</code></a>
* <a href="../../tf/nn/log_poisson_loss.md"><code>tf.nn.log_poisson_loss(targets, log_input, compute_full_loss=False, name=None)</code></a>
* <a href="../../tf/nn/log_softmax.md"><code>tf.nn.log_softmax(logits, axis=None, name=None)</code></a>
* <a href="../../tf/nn/max_pool.md"><code>tf.nn.max_pool(input, ksize, strides, padding, data_format=None, name=None)</code></a>
* `tf.nn.max_pool1d(input, ksize, strides, padding, data_format='NWC', name=None)`
* `tf.nn.max_pool2d(input, ksize, strides, padding, data_format='NHWC', name=None)`
* `tf.nn.max_pool3d(input, ksize, strides, padding, data_format='NDHWC', name=None)`
* `tf.nn.max_pool_with_argmax(input, ksize, strides, padding, data_format='NHWC', output_dtype=tf.int64, include_batch_in_index=False, name=None)`
* <a href="../../tf/nn/moments.md"><code>tf.nn.moments(x, axes, shift=None, keepdims=False, name=None)</code></a>
* `tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, num_true=1, sampled_values=None, remove_accidental_hits=False, name='nce_loss')`
* <a href="../../tf/nn/normalize_moments.md"><code>tf.nn.normalize_moments(counts, mean_ss, variance_ss, shift, name=None)</code></a>
* `tf.nn.pool(input, window_shape, pooling_type, strides=None, padding='VALID', data_format=None, dilations=None, name=None)`
* <a href="../../tf/nn/relu.md"><code>tf.nn.relu(features, name=None)</code></a>
* <a href="../../tf/nn/relu6.md"><code>tf.nn.relu6(features, name=None)</code></a>
* `tf.nn.safe_embedding_lookup_sparse(embedding_weights, sparse_ids, sparse_weights=None, combiner='mean', default_id=None, max_norm=None, name=None)`
* `tf.nn.sampled_softmax_loss(weights, biases, labels, inputs, num_sampled, num_classes, num_true=1, sampled_values=None, remove_accidental_hits=True, seed=None, name='sampled_softmax_loss')`
* <a href="../../tf/nn/scale_regularization_loss.md"><code>tf.nn.scale_regularization_loss(regularization_loss)</code></a>
* <a href="../../tf/nn/selu.md"><code>tf.nn.selu(features, name=None)</code></a>
* <a href="../../tf/nn/separable_conv2d.md"><code>tf.nn.separable_conv2d(input, depthwise_filter, pointwise_filter, strides, padding, data_format=None, dilations=None, name=None)</code></a>
* <a href="../../tf/nn/sigmoid_cross_entropy_with_logits.md"><code>tf.nn.sigmoid_cross_entropy_with_logits(labels=None, logits=None, name=None)</code></a>
* <a href="../../tf/nn/silu.md"><code>tf.nn.silu(features)</code></a>
* <a href="../../tf/nn/softmax.md"><code>tf.nn.softmax(logits, axis=None, name=None)</code></a>
* `tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis=-1, name=None)`
* <a href="../../tf/nn/softsign.md"><code>tf.nn.softsign(features, name=None)</code></a>
* `tf.nn.space_to_depth(input, block_size, data_format='NHWC', name=None)`
* <a href="../../tf/nn/sparse_softmax_cross_entropy_with_logits.md"><code>tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name=None)</code></a>
* <a href="../../tf/nn/sufficient_statistics.md"><code>tf.nn.sufficient_statistics(x, axes, shift=None, keepdims=False, name=None)</code></a>
* <a href="../../tf/nn/weighted_cross_entropy_with_logits.md"><code>tf.nn.weighted_cross_entropy_with_logits(labels, logits, pos_weight, name=None)</code></a>
* <a href="../../tf/nn/weighted_moments.md"><code>tf.nn.weighted_moments(x, axes, frequency_weights, keepdims=False, name=None)</code></a>
* <a href="../../tf/nn/with_space_to_batch.md"><code>tf.nn.with_space_to_batch(input, dilation_rate, padding, op, filter_shape=None, spatial_dims=None, data_format=None)</code></a>
* <a href="../../tf/no_op.md"><code>tf.no_op(name=None)</code></a>
* `tf.norm(tensor, ord='euclidean', axis=None, keepdims=None, name=None)`
* <a href="../../tf/numpy_function.md"><code>tf.numpy_function(func, inp, Tout, name=None)</code></a>
* <a href="../../tf/one_hot.md"><code>tf.one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None)</code></a>
* <a href="../../tf/ones.md"><code>tf.ones(shape, dtype=tf.float32, name=None)</code></a>
* <a href="../../tf/ones_like.md"><code>tf.ones_like(input, dtype=None, name=None)</code></a>
* `tf.pad(tensor, paddings, mode='CONSTANT', constant_values=0, name=None)`
* `tf.parallel_stack(values, name='parallel_stack')`
* <a href="../../tf/py_function.md"><code>tf.py_function(func, inp, Tout, name=None)</code></a>
* `tf.quantization.dequantize(input, min_range, max_range, mode='MIN_COMBINED', name=None, axis=None, narrow_range=False, dtype=tf.float32)`
* `tf.quantization.fake_quant_with_min_max_args(inputs, min=-6, max=6, num_bits=8, narrow_range=False, name=None)`
* `tf.quantization.fake_quant_with_min_max_args_gradient(gradients, inputs, min=-6, max=6, num_bits=8, narrow_range=False, name=None)`
* <a href="../../tf/quantization/fake_quant_with_min_max_vars.md"><code>tf.quantization.fake_quant_with_min_max_vars(inputs, min, max, num_bits=8, narrow_range=False, name=None)</code></a>
* <a href="../../tf/quantization/fake_quant_with_min_max_vars_gradient.md"><code>tf.quantization.fake_quant_with_min_max_vars_gradient(gradients, inputs, min, max, num_bits=8, narrow_range=False, name=None)</code></a>
* <a href="../../tf/quantization/fake_quant_with_min_max_vars_per_channel.md"><code>tf.quantization.fake_quant_with_min_max_vars_per_channel(inputs, min, max, num_bits=8, narrow_range=False, name=None)</code></a>
* <a href="../../tf/quantization/fake_quant_with_min_max_vars_per_channel_gradient.md"><code>tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient(gradients, inputs, min, max, num_bits=8, narrow_range=False, name=None)</code></a>
* `tf.quantization.quantize(input, min_range, max_range, T, mode='MIN_COMBINED', round_mode='HALF_AWAY_FROM_ZERO', name=None, narrow_range=False, axis=None, ensure_minimum_range=0.01)`
* `tf.quantization.quantize_and_dequantize(input, input_min, input_max, signed_input=True, num_bits=8, range_given=False, round_mode='HALF_TO_EVEN', name=None, narrow_range=False, axis=None)`
* `tf.quantization.quantize_and_dequantize_v2(input, input_min, input_max, signed_input=True, num_bits=8, range_given=False, round_mode='HALF_TO_EVEN', name=None, narrow_range=False, axis=None)`
* <a href="../../tf/quantization/quantized_concat.md"><code>tf.quantization.quantized_concat(concat_dim, values, input_mins, input_maxes, name=None)</code></a>
* <a href="../../tf/ragged/boolean_mask.md"><code>tf.ragged.boolean_mask(data, mask, name=None)</code></a>
* <a href="../../tf/ragged/constant.md"><code>tf.ragged.constant(pylist, dtype=None, ragged_rank=None, inner_shape=None, name=None, row_splits_dtype=tf.int64)</code></a>
* <a href="../../tf/ragged/cross.md"><code>tf.ragged.cross(inputs, name=None)</code></a>
* <a href="../../tf/ragged/cross_hashed.md"><code>tf.ragged.cross_hashed(inputs, num_buckets=0, hash_key=None, name=None)</code></a>
* <a href="../../tf/ragged/range.md"><code>tf.ragged.range(starts, limits=None, deltas=1, dtype=None, name=None, row_splits_dtype=tf.int64)</code></a>
* <a href="../../tf/ragged/row_splits_to_segment_ids.md"><code>tf.ragged.row_splits_to_segment_ids(splits, name=None, out_type=None)</code></a>
* <a href="../../tf/ragged/segment_ids_to_row_splits.md"><code>tf.ragged.segment_ids_to_row_splits(segment_ids, num_segments=None, out_type=None, name=None)</code></a>
* `tf.ragged.stack(values: List[Union[tensorflow.python.ops.ragged.ragged_tensor.RaggedTensor, tensorflow.python.ops.ragged.ragged_tensor_value.RaggedTensorValue, tensorflow.python.types.core.Tensor, tensorflow.python.types.core.TensorProtocol, int, float, bool, str, bytes, complex, tuple, list, numpy.ndarray, numpy.generic]], axis=0, name=None)`
* <a href="../../tf/ragged/stack_dynamic_partitions.md"><code>tf.ragged.stack_dynamic_partitions(data, partitions, num_partitions, name=None)</code></a>
* <a href="../../tf/random/categorical.md"><code>tf.random.categorical(logits, num_samples, dtype=None, seed=None, name=None)</code></a>
* `tf.random.experimental.stateless_fold_in(seed, data, alg='auto_select')`
* `tf.random.experimental.stateless_split(seed, num=2, alg='auto_select')`
* `tf.random.fixed_unigram_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, vocab_file='', distortion=1.0, num_reserved_ids=0, num_shards=1, shard=0, unigrams=(), seed=None, name=None)`
* <a href="../../tf/random/gamma.md"><code>tf.random.gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)</code></a>
* <a href="../../tf/random/learned_unigram_candidate_sampler.md"><code>tf.random.learned_unigram_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None)</code></a>
* <a href="../../tf/random/log_uniform_candidate_sampler.md"><code>tf.random.log_uniform_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None)</code></a>
* <a href="../../tf/random/normal.md"><code>tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)</code></a>
* <a href="../../tf/random/poisson.md"><code>tf.random.poisson(shape, lam, dtype=tf.float32, seed=None, name=None)</code></a>
* <a href="../../tf/random/shuffle.md"><code>tf.random.shuffle(value, seed=None, name=None)</code></a>
* <a href="../../tf/random/stateless_binomial.md"><code>tf.random.stateless_binomial(shape, seed, counts, probs, output_dtype=tf.int32, name=None)</code></a>
* <a href="../../tf/random/stateless_categorical.md"><code>tf.random.stateless_categorical(logits, num_samples, seed, dtype=tf.int64, name=None)</code></a>
* <a href="../../tf/random/stateless_gamma.md"><code>tf.random.stateless_gamma(shape, seed, alpha, beta=None, dtype=tf.float32, name=None)</code></a>
* `tf.random.stateless_normal(shape, seed, mean=0.0, stddev=1.0, dtype=tf.float32, name=None, alg='auto_select')`
* `tf.random.stateless_parameterized_truncated_normal(shape, seed, means=0.0, stddevs=1.0, minvals=-2.0, maxvals=2.0, name=None)`
* <a href="../../tf/random/stateless_poisson.md"><code>tf.random.stateless_poisson(shape, seed, lam, dtype=tf.int32, name=None)</code></a>
* `tf.random.stateless_truncated_normal(shape, seed, mean=0.0, stddev=1.0, dtype=tf.float32, name=None, alg='auto_select')`
* `tf.random.stateless_uniform(shape, seed, minval=0, maxval=None, dtype=tf.float32, name=None, alg='auto_select')`
* <a href="../../tf/random/truncated_normal.md"><code>tf.random.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)</code></a>
* <a href="../../tf/random/uniform.md"><code>tf.random.uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)</code></a>
* <a href="../../tf/random/uniform_candidate_sampler.md"><code>tf.random.uniform_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None)</code></a>
* `tf.range(start, limit=None, delta=1, dtype=None, name='range')`
* <a href="../../tf/rank.md"><code>tf.rank(input, name=None)</code></a>
* <a href="../../tf/realdiv.md"><code>tf.realdiv(x, y, name=None)</code></a>
* <a href="../../tf/repeat.md"><code>tf.repeat(input, repeats, axis=None, name=None)</code></a>
* <a href="../../tf/reshape.md"><code>tf.reshape(tensor, shape, name=None)</code></a>
* <a href="../../tf/reverse.md"><code>tf.reverse(tensor, axis, name=None)</code></a>
* <a href="../../tf/reverse_sequence.md"><code>tf.reverse_sequence(input, seq_lengths, seq_axis=None, batch_axis=None, name=None)</code></a>
* <a href="../../tf/roll.md"><code>tf.roll(input, shift, axis, name=None)</code></a>
* <a href="../../tf/scan.md"><code>tf.scan(fn, elems, initializer=None, parallel_iterations=10, back_prop=True, swap_memory=False, infer_shape=True, reverse=False, name=None)</code></a>
* <a href="../../tf/scatter_nd.md"><code>tf.scatter_nd(indices, updates, shape, name=None)</code></a>
* `tf.searchsorted(sorted_sequence, values, side='left', out_type=tf.int32, name=None)`
* <a href="../../tf/sequence_mask.md"><code>tf.sequence_mask(lengths, maxlen=None, dtype=tf.bool, name=None)</code></a>
* <a href="../../tf/sets/difference.md"><code>tf.sets.difference(a, b, aminusb=True, validate_indices=True)</code></a>
* <a href="../../tf/sets/intersection.md"><code>tf.sets.intersection(a, b, validate_indices=True)</code></a>
* <a href="../../tf/sets/size.md"><code>tf.sets.size(a, validate_indices=True)</code></a>
* <a href="../../tf/sets/union.md"><code>tf.sets.union(a, b, validate_indices=True)</code></a>
* <a href="../../tf/shape.md"><code>tf.shape(input, out_type=tf.int32, name=None)</code></a>
* <a href="../../tf/shape_n.md"><code>tf.shape_n(input, out_type=tf.int32, name=None)</code></a>
* `tf.signal.dct(input, type=2, n=None, axis=-1, norm=None, name=None)`
* <a href="../../tf/signal/fft.md"><code>tf.signal.fft(input, name=None)</code></a>
* <a href="../../tf/signal/fft2d.md"><code>tf.signal.fft2d(input, name=None)</code></a>
* <a href="../../tf/signal/fft3d.md"><code>tf.signal.fft3d(input, name=None)</code></a>
* <a href="../../tf/signal/fftshift.md"><code>tf.signal.fftshift(x, axes=None, name=None)</code></a>
* `tf.signal.frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1, name=None)`
* <a href="../../tf/signal/hamming_window.md"><code>tf.signal.hamming_window(window_length, periodic=True, dtype=tf.float32, name=None)</code></a>
* <a href="../../tf/signal/hann_window.md"><code>tf.signal.hann_window(window_length, periodic=True, dtype=tf.float32, name=None)</code></a>
* `tf.signal.idct(input, type=2, n=None, axis=-1, norm=None, name=None)`
* <a href="../../tf/signal/ifft.md"><code>tf.signal.ifft(input, name=None)</code></a>
* <a href="../../tf/signal/ifft2d.md"><code>tf.signal.ifft2d(input, name=None)</code></a>
* <a href="../../tf/signal/ifft3d.md"><code>tf.signal.ifft3d(input, name=None)</code></a>
* <a href="../../tf/signal/ifftshift.md"><code>tf.signal.ifftshift(x, axes=None, name=None)</code></a>
* `tf.signal.inverse_mdct(mdcts, window_fn=<function vorbis_window at 0x7fead28243a0>, norm=None, name=None)`
* `tf.signal.inverse_stft(stfts, frame_length, frame_step, fft_length=None, window_fn=<function hann_window at 0x7fead2824550>, name=None)`
* `tf.signal.inverse_stft_window_fn(frame_step, forward_window_fn=<function hann_window at 0x7fead2824550>, name=None)`
* <a href="../../tf/signal/irfft.md"><code>tf.signal.irfft(input_tensor, fft_length=None, name=None)</code></a>
* <a href="../../tf/signal/irfft2d.md"><code>tf.signal.irfft2d(input_tensor, fft_length=None, name=None)</code></a>
* <a href="../../tf/signal/irfft3d.md"><code>tf.signal.irfft3d(input_tensor, fft_length=None, name=None)</code></a>
* <a href="../../tf/signal/kaiser_bessel_derived_window.md"><code>tf.signal.kaiser_bessel_derived_window(window_length, beta=12.0, dtype=tf.float32, name=None)</code></a>
* <a href="../../tf/signal/kaiser_window.md"><code>tf.signal.kaiser_window(window_length, beta=12.0, dtype=tf.float32, name=None)</code></a>
* <a href="../../tf/signal/linear_to_mel_weight_matrix.md"><code>tf.signal.linear_to_mel_weight_matrix(num_mel_bins=20, num_spectrogram_bins=129, sample_rate=8000, lower_edge_hertz=125.0, upper_edge_hertz=3800.0, dtype=tf.float32, name=None)</code></a>
* `tf.signal.mdct(signals, frame_length, window_fn=<function vorbis_window at 0x7fead28243a0>, pad_end=False, norm=None, name=None)`
* <a href="../../tf/signal/mfccs_from_log_mel_spectrograms.md"><code>tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms, name=None)</code></a>
* <a href="../../tf/signal/overlap_and_add.md"><code>tf.signal.overlap_and_add(signal, frame_step, name=None)</code></a>
* <a href="../../tf/signal/rfft.md"><code>tf.signal.rfft(input_tensor, fft_length=None, name=None)</code></a>
* <a href="../../tf/signal/rfft2d.md"><code>tf.signal.rfft2d(input_tensor, fft_length=None, name=None)</code></a>
* <a href="../../tf/signal/rfft3d.md"><code>tf.signal.rfft3d(input_tensor, fft_length=None, name=None)</code></a>
* `tf.signal.stft(signals, frame_length, frame_step, fft_length=None, window_fn=<function hann_window at 0x7fead2824550>, pad_end=False, name=None)`
* <a href="../../tf/signal/vorbis_window.md"><code>tf.signal.vorbis_window(window_length, dtype=tf.float32, name=None)</code></a>
* <a href="../../tf/size.md"><code>tf.size(input, out_type=tf.int32, name=None)</code></a>
* <a href="../../tf/slice.md"><code>tf.slice(input_, begin, size, name=None)</code></a>
* `tf.sort(values, axis=-1, direction='ASCENDING', name=None)`
* <a href="../../tf/space_to_batch.md"><code>tf.space_to_batch(input, block_shape, paddings, name=None)</code></a>
* <a href="../../tf/space_to_batch_nd.md"><code>tf.space_to_batch_nd(input, block_shape, paddings, name=None)</code></a>
* `tf.split(value, num_or_size_splits, axis=0, num=None, name='split')`
* <a href="../../tf/squeeze.md"><code>tf.squeeze(input, axis=None, name=None)</code></a>
* `tf.stack(values, axis=0, name='stack')`
* <a href="../../tf/stop_gradient.md"><code>tf.stop_gradient(input, name=None)</code></a>
* <a href="../../tf/strided_slice.md"><code>tf.strided_slice(input_, begin, end, strides=None, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0, var=None, name=None)</code></a>
* `tf.strings.as_string(input, precision=-1, scientific=False, shortest=False, width=-1, fill='', name=None)`
* <a href="../../tf/strings/bytes_split.md"><code>tf.strings.bytes_split(input, name=None)</code></a>
* `tf.strings.format(template, inputs, placeholder='{}', summarize=3, name=None)`
* `tf.strings.join(inputs, separator='', name=None)`
* `tf.strings.length(input, unit='BYTE', name=None)`
* `tf.strings.lower(input, encoding='', name=None)`
* `tf.strings.ngrams(data, ngram_width, separator=' ', pad_values=None, padding_width=None, preserve_short_sequences=False, name=None)`
* `tf.strings.reduce_join(inputs, axis=None, keepdims=False, separator='', name=None)`
* <a href="../../tf/strings/regex_full_match.md"><code>tf.strings.regex_full_match(input, pattern, name=None)</code></a>
* <a href="../../tf/strings/regex_replace.md"><code>tf.strings.regex_replace(input, pattern, rewrite, replace_global=True, name=None)</code></a>
* `tf.strings.split(input, sep=None, maxsplit=-1, name=None)`
* <a href="../../tf/strings/strip.md"><code>tf.strings.strip(input, name=None)</code></a>
* `tf.strings.substr(input, pos, len, unit='BYTE', name=None)`
* <a href="../../tf/strings/to_hash_bucket.md"><code>tf.strings.to_hash_bucket(input, num_buckets, name=None)</code></a>
* <a href="../../tf/strings/to_hash_bucket_fast.md"><code>tf.strings.to_hash_bucket_fast(input, num_buckets, name=None)</code></a>
* <a href="../../tf/strings/to_hash_bucket_strong.md"><code>tf.strings.to_hash_bucket_strong(input, num_buckets, key, name=None)</code></a>
* <a href="../../tf/strings/to_number.md"><code>tf.strings.to_number(input, out_type=tf.float32, name=None)</code></a>
* `tf.strings.unicode_decode(input, input_encoding, errors='replace', replacement_char=65533, replace_control_characters=False, name=None)`
* `tf.strings.unicode_decode_with_offsets(input, input_encoding, errors='replace', replacement_char=65533, replace_control_characters=False, name=None)`
* `tf.strings.unicode_encode(input, output_encoding, errors='replace', replacement_char=65533, name=None)`
* <a href="../../tf/strings/unicode_script.md"><code>tf.strings.unicode_script(input, name=None)</code></a>
* `tf.strings.unicode_split(input, input_encoding, errors='replace', replacement_char=65533, name=None)`
* `tf.strings.unicode_split_with_offsets(input, input_encoding, errors='replace', replacement_char=65533, name=None)`
* `tf.strings.unicode_transcode(input, input_encoding, output_encoding, errors='replace', replacement_char=65533, replace_control_characters=False, name=None)`
* `tf.strings.unsorted_segment_join(inputs, segment_ids, num_segments, separator='', name=None)`
* `tf.strings.upper(input, encoding='', name=None)`
* <a href="../../tf/tensor_scatter_nd_add.md"><code>tf.tensor_scatter_nd_add(tensor, indices, updates, name=None)</code></a>
* <a href="../../tf/tensor_scatter_nd_max.md"><code>tf.tensor_scatter_nd_max(tensor, indices, updates, name=None)</code></a>
* <a href="../../tf/tensor_scatter_nd_min.md"><code>tf.tensor_scatter_nd_min(tensor, indices, updates, name=None)</code></a>
* <a href="../../tf/tensor_scatter_nd_sub.md"><code>tf.tensor_scatter_nd_sub(tensor, indices, updates, name=None)</code></a>
* <a href="../../tf/tensor_scatter_nd_update.md"><code>tf.tensor_scatter_nd_update(tensor, indices, updates, name=None)</code></a>
* <a href="../../tf/tensordot.md"><code>tf.tensordot(a, b, axes, name=None)</code></a>
* <a href="../../tf/tile.md"><code>tf.tile(input, multiples, name=None)</code></a>
* <a href="../../tf/timestamp.md"><code>tf.timestamp(name=None)</code></a>
* `tf.transpose(a, perm=None, conjugate=False, name='transpose')`
* <a href="../../tf/truncatediv.md"><code>tf.truncatediv(x, y, name=None)</code></a>
* <a href="../../tf/truncatemod.md"><code>tf.truncatemod(x, y, name=None)</code></a>
* <a href="../../tf/tuple.md"><code>tf.tuple(tensors, control_inputs=None, name=None)</code></a>
* <a href="../../tf/unique.md"><code>tf.unique(x, out_idx=tf.int32, name=None)</code></a>
* <a href="../../tf/unique_with_counts.md"><code>tf.unique_with_counts(x, out_idx=tf.int32, name=None)</code></a>
* <a href="../../tf/unravel_index.md"><code>tf.unravel_index(indices, dims, name=None)</code></a>
* `tf.unstack(value, num=None, axis=0, name='unstack')`
* <a href="../../tf/where.md"><code>tf.where(condition, x=None, y=None, name=None)</code></a>
* `tf.xla_all_reduce(input, group_assignment, reduce_op, name=None)`
* `tf.xla_broadcast_helper(lhs, rhs, broadcast_dims, name=None)`
* `tf.xla_cluster_output(input, name=None)`
* `tf.xla_conv(lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count, dimension_numbers, precision_config, name=None)`
* `tf.xla_conv_v2(lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count, dimension_numbers, precision_config, preferred_element_type, name=None)`
* `tf.xla_dequantize(input, min_range, max_range, mode, transpose_output, name=None)`
* `tf.xla_dot(lhs, rhs, dimension_numbers, precision_config, name=None)`
* `tf.xla_dot_v2(lhs, rhs, dimension_numbers, precision_config, preferred_element_type, name=None)`
* `tf.xla_dynamic_slice(input, start_indices, size_indices, name=None)`
* `tf.xla_dynamic_update_slice(input, update, indices, name=None)`
* `tf.xla_einsum(a, b, equation, name=None)`
* `tf.xla_gather(operand, start_indices, slice_sizes, dimension_numbers, indices_are_sorted, name=None)`
* `tf.xla_if(cond, inputs, then_branch, else_branch, Tout, name=None)`
* `tf.xla_key_value_sort(keys, values, name=None)`
* `tf.xla_launch(constants, args, resources, Tresults, function, name=None)`
* `tf.xla_pad(input, padding_value, padding_low, padding_high, padding_interior, name=None)`
* `tf.xla_recv(dtype, tensor_name, shape, name=None)`
* `tf.xla_reduce(input, init_value, dimensions_to_reduce, reducer, name=None)`
* `tf.xla_reduce_scatter(input, group_assignment, scatter_dimension, reduce_op, name=None)`
* `tf.xla_reduce_window(input, init_value, window_dimensions, window_strides, base_dilations, window_dilations, padding, computation, name=None)`
* `tf.xla_remove_dynamic_dimension_size(input, dim_index, name=None)`
* `tf.xla_replica_id(name=None)`
* `tf.xla_rng_bit_generator(algorithm, initial_state, shape, dtype=tf.uint64, name=None)`
* `tf.xla_scatter(operand, scatter_indices, updates, update_computation, dimension_numbers, indices_are_sorted, name=None)`
* `tf.xla_select_and_scatter(operand, window_dimensions, window_strides, padding, source, init_value, select, scatter, name=None)`
* `tf.xla_self_adjoint_eig(a, lower, max_iter, epsilon, name=None)`
* `tf.xla_send(tensor, tensor_name, name=None)`
* `tf.xla_set_bound(input, bound, name=None)`
* `tf.xla_set_dynamic_dimension_size(input, dim_index, size, name=None)`
* `tf.xla_sharding(input, sharding='', unspecified_dims=[], name=None)`
* `tf.xla_sort(input, name=None)`
* `tf.xla_spmd_full_to_shard_shape(input, manual_sharding, dim=-1, unspecified_dims=[], name=None)`
* `tf.xla_spmd_shard_to_full_shape(input, manual_sharding, full_shape, dim=-1, unspecified_dims=[], name=None)`
* `tf.xla_svd(a, max_iter, epsilon, precision_config, name=None)`
* `tf.xla_variadic_reduce(input, init_value, dimensions_to_reduce, reducer, name=None)`
* `tf.xla_variadic_reduce_v2(inputs, init_values, dimensions_to_reduce, reducer, name=None)`
* `tf.xla_variadic_sort(inputs, dimension, comparator, is_stable, name=None)`
* `tf.xla_while(input, cond, body, name=None)`
* <a href="../../tf/zeros.md"><code>tf.zeros(shape, dtype=tf.float32, name=None)</code></a>
* <a href="../../tf/zeros_like.md"><code>tf.zeros_like(input, dtype=None, name=None)</code></a>