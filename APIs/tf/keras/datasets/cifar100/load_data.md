description: Loads the CIFAR100 dataset.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.datasets.cifar100.load_data" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.datasets.cifar100.load_data

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/datasets/cifar100.py#L27-L94">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Loads the CIFAR100 dataset.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.datasets.cifar100.load_data`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.datasets.cifar100.load_data(
    label_mode=&#x27;fine&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is a dataset of 50,000 32x32 color training images and
10,000 test images, labeled over 100 fine-grained classes that are
grouped into 20 coarse-grained classes. See more info at the
[CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`label_mode`
</td>
<td>
one of "fine", "coarse". If it is "fine" the category labels
are the fine-grained labels, if it is "coarse" the output labels are the
coarse-grained superclasses.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.
</td>
</tr>

</table>


**x_train**: uint8 NumPy array of grayscale image data with shapes
  `(50000, 32, 32, 3)`, containing the training data. Pixel values range
  from 0 to 255.

**y_train**: uint8 NumPy array of labels (integers in range 0-99)
  with shape `(50000, 1)` for the training data.

**x_test**: uint8 NumPy array of grayscale image data with shapes
  `(10000, 32, 32, 3)`, containing the test data. Pixel values range
  from 0 to 255.

**y_test**: uint8 NumPy array of labels (integers in range 0-99)
  with shape `(10000, 1)` for the test data.

#### Example:



```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)
```