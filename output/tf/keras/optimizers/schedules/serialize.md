description: Serializes a LearningRateSchedule into a JSON-compatible representation.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.optimizers.schedules.serialize" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.optimizers.schedules.serialize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.7.0/keras/optimizer_v2/learning_rate_schedule.py#L1032-L1054">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Serializes a `LearningRateSchedule` into a JSON-compatible representation.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.optimizers.schedules.serialize`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.optimizers.schedules.serialize`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.optimizers.schedules.serialize(
    learning_rate_schedule
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`learning_rate_schedule`
</td>
<td>
The `LearningRateSchedule` object to serialize.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A JSON-serializable dict representing the object's config.
</td>
</tr>

</table>



#### Example:



```
>>> lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
...   0.1, decay_steps=100000, decay_rate=0.96, staircase=True)
>>> tf.keras.optimizers.schedules.serialize(lr_schedule)
{'class_name': 'ExponentialDecay',
 'config': {'decay_rate': 0.96,
    'decay_steps': 100000,
    'initial_learning_rate': 0.1,
    'name': None,
    'staircase': True}}
```