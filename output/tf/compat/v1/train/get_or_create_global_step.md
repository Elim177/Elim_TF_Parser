description: Returns and create (if necessary) the global step tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.train.get_or_create_global_step" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.train.get_or_create_global_step

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/training/training_util.py">View source</a>



Returns and create (if necessary) the global step tensor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.train.get_or_create_global_step(
    graph=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

With the deprecation of global graphs, TF no longer tracks variables in
collections. In other words, there are no global variables in TF2. Thus, the
global step functions have been removed  (`get_or_create_global_step`,
`create_global_step`, `get_global_step`) . You have two options for migrating:

1. Create a Keras optimizer, which generates an `iterations` variable. This
   variable is automatically incremented when calling `apply_gradients`.
2. Manually create and increment a <a href="../../../../tf/Variable.md"><code>tf.Variable</code></a>.

Below is an example of migrating away from using a global step to using a
Keras optimizer:

Define a dummy model and loss:

```
>>> def compute_loss(x):
...   v = tf.Variable(3.0)
...   y = x * v
...   loss = x * 5 - x * v
...   return loss, [v]
```

Before migrating:

```
>>> g = tf.Graph()
>>> with g.as_default():
...   x = tf.compat.v1.placeholder(tf.float32, [])
...   loss, var_list = compute_loss(x)
...   global_step = tf.compat.v1.train.get_or_create_global_step()
...   global_init = tf.compat.v1.global_variables_initializer()
...   optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
...   train_op = optimizer.minimize(loss, global_step, var_list)
>>> sess = tf.compat.v1.Session(graph=g)
>>> sess.run(global_init)
>>> print("before training:", sess.run(global_step))
before training: 0
>>> sess.run(train_op, feed_dict={x: 3})
>>> print("after training:", sess.run(global_step))
after training: 1
```

Migrating to a Keras optimizer:

```
>>> optimizer = tf.keras.optimizers.SGD(.01)
>>> print("before training:", optimizer.iterations.numpy())
before training: 0
>>> with tf.GradientTape() as tape:
...   loss, var_list = compute_loss(3)
...   grads = tape.gradient(loss, var_list)
...   optimizer.apply_gradients(zip(grads, var_list))
>>> print("after training:", optimizer.iterations.numpy())
after training: 1
```



 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`graph`
</td>
<td>
The graph in which to create the global step tensor. If missing, use
default graph.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The global step tensor.
</td>
</tr>

</table>


