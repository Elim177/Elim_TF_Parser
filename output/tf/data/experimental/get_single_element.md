description: Returns the single element of the dataset as a nested structure of tensors. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.get_single_element" />
<meta itemprop="path" content="Stable" />
</div>

# tf.data.experimental.get_single_element

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/data/experimental/ops/get_single_element.py">View source</a>



Returns the single element of the `dataset` as a nested structure of tensors. (deprecated)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.data.experimental.get_single_element`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.get_single_element(
    dataset
)
</code></pre>



<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use <a href="../../../tf/data/Dataset.md#get_single_element"><code>tf.data.Dataset.get_single_element()</code></a>.

The function enables you to use a <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> in a stateless
"tensor-in tensor-out" expression, without creating an iterator.
This facilitates the ease of data transformation on tensors using the
optimized <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> abstraction on top of them.

For example, lets consider a `preprocessing_fn` which would take as an
input the raw features and returns the processed feature along with
it's label.

```python
def preprocessing_fn(raw_feature):
  # ... the raw_feature is preprocessed as per the use-case
  return feature

raw_features = ...  # input batch of BATCH_SIZE elements.
dataset = (tf.data.Dataset.from_tensor_slices(raw_features)
           .map(preprocessing_fn, num_parallel_calls=BATCH_SIZE)
           .batch(BATCH_SIZE))

processed_features = tf.data.experimental.get_single_element(dataset)
```

In the above example, the `raw_features` tensor of length=BATCH_SIZE
was converted to a <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a>. Next, each of the `raw_feature` was
mapped using the `preprocessing_fn` and the processed features were
grouped into a single batch. The final `dataset` contains only one element
which is a batch of all the processed features.

NOTE: The `dataset` should contain only one element.

Now, instead of creating an iterator for the `dataset` and retrieving the
batch of features, the <a href="../../../tf/data/experimental/get_single_element.md"><code>tf.data.experimental.get_single_element()</code></a> function
is used to skip the iterator creation process and directly output the batch
of features.

This can be particularly useful when your tensor transformations are
expressed as <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> operations, and you want to use those
transformations while serving your model.

# Keras

```python

model = ... # A pre-built or custom model

class PreprocessingModel(tf.keras.Model):
  def __init__(self, model):
    super().__init__(self)
    self.model = model

  @tf.function(input_signature=[...])
  def serving_fn(self, data):
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.map(preprocessing_fn, num_parallel_calls=BATCH_SIZE)
    ds = ds.batch(batch_size=BATCH_SIZE)
    return tf.argmax(
      self.model(tf.data.experimental.get_single_element(ds)),
      axis=-1
    )

preprocessing_model = PreprocessingModel(model)
your_exported_model_dir = ... # save the model to this path.
tf.saved_model.save(preprocessing_model, your_exported_model_dir,
              signatures={'serving_default': preprocessing_model.serving_fn})
```

# Estimator

In the case of estimators, you need to generally define a `serving_input_fn`
which would require the features to be processed by the model while
inferencing.

```python
def serving_input_fn():

  raw_feature_spec = ... # Spec for the raw_features
  input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      raw_feature_spec, default_batch_size=None)
  )
  serving_input_receiver = input_fn()
  raw_features = serving_input_receiver.features

  def preprocessing_fn(raw_feature):
    # ... the raw_feature is preprocessed as per the use-case
    return feature

  dataset = (tf.data.Dataset.from_tensor_slices(raw_features)
            .map(preprocessing_fn, num_parallel_calls=BATCH_SIZE)
            .batch(BATCH_SIZE))

  processed_features = tf.data.experimental.get_single_element(dataset)

  # Please note that the value of `BATCH_SIZE` should be equal to
  # the size of the leading dimension of `raw_features`. This ensures
  # that `dataset` has only element, which is a pre-requisite for
  # using `tf.data.experimental.get_single_element(dataset)`.

  return tf.estimator.export.ServingInputReceiver(
      processed_features, serving_input_receiver.receiver_tensors)

estimator = ... # A pre-built or custom estimator
estimator.export_saved_model(your_exported_model_dir, serving_input_fn)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dataset`
</td>
<td>
A <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> object containing a single element.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A nested structure of <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a> objects, corresponding to the single
element of `dataset`.
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
if `dataset` is not a <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> object.
</td>
</tr><tr>
<td>
`InvalidArgumentError`
</td>
<td>
(at runtime) if `dataset` does not contain exactly
one element.
</td>
</tr>
</table>

