description: Decode a 16-bit PCM WAV file to a float tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.audio.decode_wav" />
<meta itemprop="path" content="Stable" />
<div/>

# tf.audio.decode_wav

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Decode a 16-bit PCM WAV file to a float tensor.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.audio.decode_wav`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.audio.decode_wav(
    contents, desired_channels=-1, desired_samples=-1, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The -32768 to 32767 signed 16-bit values will be scaled to -1.0 to 1.0 in float.

When desired_channels is set, if the input contains fewer channels than this
then the last channel will be duplicated to give the requested number, else if
the input has more channels than requested then the additional channels will be
ignored.

If desired_samples is set, then the audio will be cropped or padded with zeroes
to the requested length.

The first output contains a Tensor with the content of the audio samples. The
lowest dimension will be the number of channels, and the second will be the
number of samples. For example, a ten-sample-long stereo WAV file should give an
output shape of [10, 2].

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`contents`
</td>
<td>
A `Tensor` of type `string`.
The WAV-encoded audio, usually from a file.
</td>
</tr><tr>
<td>
`desired_channels`
</td>
<td>
An optional `int`. Defaults to `-1`.
Number of sample channels wanted.
</td>
</tr><tr>
<td>
`desired_samples`
</td>
<td>
An optional `int`. Defaults to `-1`.
Length of audio requested.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tuple of `Tensor` objects (audio, sample_rate).
</td>
</tr>
<tr>
<td>
`audio`
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`sample_rate`
</td>
<td>
A `Tensor` of type `int32`.
</td>
</tr>
</table>


This guide trains a neural network model to classify images of clothing, like sneakers and shirts. It's okay if you don't understand all the details; this is a fast-paced overview of a complete TensorFlow program with the details explained as you go.
This guide uses tf.keras, a high-level API to build and train models in TensorFlow.
This guide uses the Fashion MNIST dataset which contains 70,000 grayscale images in 10 categories. The images show individual articles of clothing at low resolution (28 by 28 pixels), as seen here:
Fashion MNIST is intended as a drop-in replacement for the classic MNIST dataset—often used as the "Hello, World" of machine learning programs for computer vision. The MNIST dataset contains images of handwritten digits (0, 1, 2, etc.) in a format identical to that of the articles of clothing you'll use here.
This guide uses Fashion MNIST for variety, and because it's a slightly more challenging problem than regular MNIST. Both datasets are relatively small and are used to verify that an algorithm works as expected. They're good starting points to test and debug code.
Here, 60,000 images are used to train the network and 10,000 images to evaluate how accurately the network learned to classify images. You can access the Fashion MNIST directly from TensorFlow. Import and load the Fashion MNIST data directly from TensorFlow:
Loading the dataset returns four NumPy arrays:
The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255. The labels are an array of integers, ranging from 0 to 9. These correspond to the class of clothing the image represents:
Each image is mapped to a single label. Since the class names are not included with the dataset, store them here to use later when plotting the images:
Let's explore the format of the dataset before training the model. The following shows there are 60,000 images in the training set, with each image represented as 28 x 28 pixels:
Likewise, there are 60,000 labels in the training set:
Each label is an integer between 0 and 9:
There are 10,000 images in the test set. Again, each image is represented as 28 x 28 pixels:
And the test set contains 10,000 images labels:
The data must be preprocessed before training the network. If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255:

Scale these values to a range of 0 to 1 before feeding them to the neural network model. To do so, divide the values by 255. It's important that the training set and the testing set be preprocessed in the same way:
To verify that the data is in the correct format and that you're ready to build and train the network, let's display the first 25 images from the training set and display the class name below each image.

Building the neural network requires configuring the layers of the model, then compiling the model.
The basic building block of a neural network is the layer. Layers extract representations from the data fed into them. Hopefully, these representations are meaningful for the problem at hand.
Most of deep learning consists of chaining together simple layers. Most layers, such as tf.keras.layers.Dense, have parameters that are learned during training.
The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels). Think of this layer as unstacking rows of pixels in the image and lining them up. This layer has no parameters to learn; it only reformats the data.
After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers. These are densely connected, or fully connected, neural layers. The first Dense layer has 128 nodes (or neurons). The second (and last) layer returns a logits array with length of 10. Each node contains a score that indicates the current image belongs to one of the 10 classes.
Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:
Training the neural network model requires the following steps:
To start training,  call the model.fit method—so called because it "fits" the model to the training data:
As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of about 0.91 (or 91%) on the training data.
Next, compare how the model performs on the test dataset:
It turns out that the accuracy on the test dataset is a little less than the accuracy on the training dataset. This gap between training accuracy and test accuracy represents overfitting. Overfitting happens when a machine learning model performs worse on new, previously unseen inputs than it does on the training data. An overfitted model "memorizes" the noise and details in the training dataset to a point where it negatively impacts the performance of the model on the new data. For more information, see the following:
With the model trained, you can use it to make predictions about some images.
The model's linear outputs, logits. Attach a softmax layer to convert the logits to probabilities, which are easier to interpret. 
Here, the model has predicted the label for each image in the testing set. Let's take a look at the first prediction:
A prediction is an array of 10 numbers. They represent the model's "confidence" that the image corresponds to each of the 10 different articles of clothing. You can see which label has the highest confidence value:
So, the model is most confident that this image is an ankle boot, or class_names[9]. Examining the test label shows that this classification is correct:
Graph this to look at the full set of 10 class predictions.
With the model trained, you can use it to make predictions about some images.
Let's look at the 0th image, predictions, and prediction array. Correct prediction labels are blue and incorrect prediction labels are red. The number gives the percentage (out of 100) for the predicted label.


Let's plot several images with their predictions. Note that the model can be wrong even when very confident.

Finally, use the trained model to make a prediction about a single image.
tf.keras models are optimized to make predictions on a batch, or collection, of examples at once. Accordingly, even though you're using a single image, you need to add it to a list:
Now predict the correct label for this image:

tf.keras.Model.predict returns a list of lists—one list for each image in the batch of data. Grab the predictions for our (only) image in the batch:
And the model predicts a label as expected.
Except as otherwise noted, the content of this page is licensed under the Creative Commons Attribution 4.0 License, and code samples are licensed under the Apache 2.0 License. For details, see the Google Developers Site Policies. Java is a registered trademark of Oracle and/or its affiliates.
Last updated 2021-11-11 UTC.

This guide trains a neural network model to classify images of clothing, like sneakers and shirts. It's okay if you don't understand all the details; this is a fast-paced overview of a complete TensorFlow program with the details explained as you go.
This guide uses tf.keras, a high-level API to build and train models in TensorFlow.
This guide uses the Fashion MNIST dataset which contains 70,000 grayscale images in 10 categories. The images show individual articles of clothing at low resolution (28 by 28 pixels), as seen here:
Fashion MNIST is intended as a drop-in replacement for the classic MNIST dataset—often used as the "Hello, World" of machine learning programs for computer vision. The MNIST dataset contains images of handwritten digits (0, 1, 2, etc.) in a format identical to that of the articles of clothing you'll use here.
This guide uses Fashion MNIST for variety, and because it's a slightly more challenging problem than regular MNIST. Both datasets are relatively small and are used to verify that an algorithm works as expected. They're good starting points to test and debug code.
Here, 60,000 images are used to train the network and 10,000 images to evaluate how accurately the network learned to classify images. You can access the Fashion MNIST directly from TensorFlow. Import and load the Fashion MNIST data directly from TensorFlow:
Loading the dataset returns four NumPy arrays:
The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255. The labels are an array of integers, ranging from 0 to 9. These correspond to the class of clothing the image represents:
Each image is mapped to a single label. Since the class names are not included with the dataset, store them here to use later when plotting the images:
Let's explore the format of the dataset before training the model. The following shows there are 60,000 images in the training set, with each image represented as 28 x 28 pixels:
Likewise, there are 60,000 labels in the training set:
Each label is an integer between 0 and 9:
There are 10,000 images in the test set. Again, each image is represented as 28 x 28 pixels:
And the test set contains 10,000 images labels:
The data must be preprocessed before training the network. If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255:

Scale these values to a range of 0 to 1 before feeding them to the neural network model. To do so, divide the values by 255. It's important that the training set and the testing set be preprocessed in the same way:
To verify that the data is in the correct format and that you're ready to build and train the network, let's display the first 25 images from the training set and display the class name below each image.

Building the neural network requires configuring the layers of the model, then compiling the model.
The basic building block of a neural network is the layer. Layers extract representations from the data fed into them. Hopefully, these representations are meaningful for the problem at hand.
Most of deep learning consists of chaining together simple layers. Most layers, such as tf.keras.layers.Dense, have parameters that are learned during training.
The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels). Think of this layer as unstacking rows of pixels in the image and lining them up. This layer has no parameters to learn; it only reformats the data.
After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers. These are densely connected, or fully connected, neural layers. The first Dense layer has 128 nodes (or neurons). The second (and last) layer returns a logits array with length of 10. Each node contains a score that indicates the current image belongs to one of the 10 classes.
Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:
Training the neural network model requires the following steps:
To start training,  call the model.fit method—so called because it "fits" the model to the training data:
As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of about 0.91 (or 91%) on the training data.
Next, compare how the model performs on the test dataset:
It turns out that the accuracy on the test dataset is a little less than the accuracy on the training dataset. This gap between training accuracy and test accuracy represents overfitting. Overfitting happens when a machine learning model performs worse on new, previously unseen inputs than it does on the training data. An overfitted model "memorizes" the noise and details in the training dataset to a point where it negatively impacts the performance of the model on the new data. For more information, see the following:
With the model trained, you can use it to make predictions about some images.
The model's linear outputs, logits. Attach a softmax layer to convert the logits to probabilities, which are easier to interpret. 
Here, the model has predicted the label for each image in the testing set. Let's take a look at the first prediction:
A prediction is an array of 10 numbers. They represent the model's "confidence" that the image corresponds to each of the 10 different articles of clothing. You can see which label has the highest confidence value:
So, the model is most confident that this image is an ankle boot, or class_names[9]. Examining the test label shows that this classification is correct:
Graph this to look at the full set of 10 class predictions.
With the model trained, you can use it to make predictions about some images.
Let's look at the 0th image, predictions, and prediction array. Correct prediction labels are blue and incorrect prediction labels are red. The number gives the percentage (out of 100) for the predicted label.


Let's plot several images with their predictions. Note that the model can be wrong even when very confident.

Finally, use the trained model to make a prediction about a single image.
tf.keras models are optimized to make predictions on a batch, or collection, of examples at once. Accordingly, even though you're using a single image, you need to add it to a list:
Now predict the correct label for this image:

tf.keras.Model.predict returns a list of lists—one list for each image in the batch of data. Grab the predictions for our (only) image in the batch:
And the model predicts a label as expected.
Except as otherwise noted, the content of this page is licensed under the Creative Commons Attribution 4.0 License, and code samples are licensed under the Apache 2.0 License. For details, see the Google Developers Site Policies. Java is a registered trademark of Oracle and/or its affiliates.
Last updated 2021-11-11 UTC.
