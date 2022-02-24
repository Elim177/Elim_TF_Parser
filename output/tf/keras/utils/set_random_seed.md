description: Sets all random seeds for the program (Python, NumPy, and TensorFlow).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.set_random_seed" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.utils.set_random_seed

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/utils/tf_utils.py#L34-L63">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Sets all random seeds for the program (Python, NumPy, and TensorFlow).

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.set_random_seed(
    seed
)
</code></pre>



<!-- Placeholder for "Used in" -->

You can use this utility to make almost any Keras program fully deterministic.
Some limitations apply in cases where network communications are involved
(e.g. parameter server distribution), which creates additional sources of
randomness, or when certain non-deterministic cuDNN ops are involved.

Calling this utility is equivalent to the following:

```python
import random
import numpy as np
import tensorflow as tf
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Arguments</h2></th></tr>

<tr>
<td>
`seed`
</td>
<td>
Integer, the random seed to use.
</td>
</tr>
</table>

