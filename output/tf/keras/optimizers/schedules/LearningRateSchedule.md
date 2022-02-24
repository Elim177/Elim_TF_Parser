description: The learning rate schedule base class.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.optimizers.schedules.LearningRateSchedule" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tf.keras.optimizers.schedules.LearningRateSchedule

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/optimizer_v2/learning_rate_schedule.py#L26-L88">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The learning rate schedule base class.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.optimizers.schedules.LearningRateSchedule`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.optimizers.schedules.LearningRateSchedule`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

You can use a learning rate schedule to modulate how the learning rate
of your optimizer changes over time.

Several built-in learning rate schedules are available, such as
<a href="../../../../tf/keras/optimizers/schedules/ExponentialDecay.md"><code>tf.keras.optimizers.schedules.ExponentialDecay</code></a> or
<a href="../../../../tf/keras/optimizers/schedules/PiecewiseConstantDecay.md"><code>tf.keras.optimizers.schedules.PiecewiseConstantDecay</code></a>:

```python
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
```

A `LearningRateSchedule` instance can be passed in as the `learning_rate`
argument of any optimizer.

To implement your own schedule object, you should implement the `__call__`
method, which takes a `step` argument (scalar integer tensor, the
current training step count).
Like for any other Keras object, you can also optionally
make your object serializable by implementing the `get_config`
and `from_config` methods.

#### Example:



```python
class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, initial_learning_rate):
    self.initial_learning_rate = initial_learning_rate

  def __call__(self, step):
     return self.initial_learning_rate / (step + 1)

optimizer = tf.keras.optimizers.SGD(learning_rate=MyLRSchedule(0.1))
```

## Methods

<h3 id="from_config"><code>from_config</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/optimizer_v2/learning_rate_schedule.py#L78-L88">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_config(
    config
)
</code></pre>

Instantiates a `LearningRateSchedule` from its config.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`config`
</td>
<td>
Output of `get_config()`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `LearningRateSchedule` instance.
</td>
</tr>

</table>



<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/optimizer_v2/learning_rate_schedule.py#L74-L76">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>get_config()
</code></pre>




<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/optimizer_v2/learning_rate_schedule.py#L70-L72">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>__call__(
    step
)
</code></pre>

Call self as a function.




