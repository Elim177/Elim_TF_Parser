description: Instantiates the VGG19 architecture.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.applications.vgg19.VGG19" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.applications.vgg19.VGG19

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/applications/vgg19.py#L43-L231">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Instantiates the VGG19 architecture.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.applications.VGG19`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.applications.VGG19`, `tf.compat.v1.keras.applications.vgg19.VGG19`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.applications.vgg19.VGG19(
    include_top=(True), weights=&#x27;imagenet&#x27;, input_tensor=None,
    input_shape=None, pooling=None, classes=1000,
    classifier_activation=&#x27;softmax&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Reference:


- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)

For image classification use cases, see
[this page for detailed examples](
  https://keras.io/api/applications/#usage-examples-for-image-classification-models).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning](
  https://keras.io/guides/transfer_learning/).

The default input size for this model is 224x224.

Note: each Keras Application expects a specific kind of input preprocessing.
For VGG19, call <a href="../../../../tf/keras/applications/vgg19/preprocess_input.md"><code>tf.keras.applications.vgg19.preprocess_input</code></a> on your
inputs before passing them to the model.
<a href="../../../../tf/keras/applications/vgg19/preprocess_input.md"><code>vgg19.preprocess_input</code></a> will convert the input images from RGB to BGR,
then will zero-center each color channel with respect to the ImageNet dataset,
without scaling.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`include_top`
</td>
<td>
whether to include the 3 fully-connected
layers at the top of the network.
</td>
</tr><tr>
<td>
`weights`
</td>
<td>
one of `None` (random initialization),
'imagenet' (pre-training on ImageNet),
or the path to the weights file to be loaded.
</td>
</tr><tr>
<td>
`input_tensor`
</td>
<td>
optional Keras tensor
(i.e. output of <a href="../../../../tf/keras/Input.md"><code>layers.Input()</code></a>)
to use as image input for the model.
</td>
</tr><tr>
<td>
`input_shape`
</td>
<td>
optional shape tuple, only to be specified
if `include_top` is False (otherwise the input shape
has to be `(224, 224, 3)`
(with `channels_last` data format)
or `(3, 224, 224)` (with `channels_first` data format).
It should have exactly 3 inputs channels,
and width and height should be no smaller than 32.
E.g. `(200, 200, 3)` would be one valid value.
</td>
</tr><tr>
<td>
`pooling`
</td>
<td>
Optional pooling mode for feature extraction
when `include_top` is `False`.
- `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional block.
- `avg` means that global average pooling
    will be applied to the output of the
    last convolutional block, and thus
    the output of the model will be a 2D tensor.
- `max` means that global max pooling will
    be applied.
</td>
</tr><tr>
<td>
`classes`
</td>
<td>
optional number of classes to classify images
into, only to be specified if `include_top` is True, and
if no `weights` argument is specified.
</td>
</tr><tr>
<td>
`classifier_activation`
</td>
<td>
A `str` or callable. The activation function to use
on the "top" layer. Ignored unless `include_top=True`. Set
`classifier_activation=None` to return the logits of the "top" layer.
When loading pretrained weights, `classifier_activation` can only
be `None` or `"softmax"`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../../../tf/keras/Model.md"><code>keras.Model</code></a> instance.
</td>
</tr>

</table>

