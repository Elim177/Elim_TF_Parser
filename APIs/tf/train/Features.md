description: Protocol message for describing the features of a <a href="../../tf/train/Example.md"><code>tf.train.Example</code></a>.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.train.Features" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="FeatureEntry"/>
</div>

# tf.train.Features

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/core/example/feature.proto">View source</a>



Protocol message for describing the `features` of a <a href="../../tf/train/Example.md"><code>tf.train.Example</code></a>.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.train.Features`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

`Features` are organized into categories by name.  The `Features` message
contains the mapping from name to <a href="../../tf/train/Feature.md"><code>tf.train.Feature</code></a>.

One item value of `Features` for a movie recommendation application:

```
    feature {
      key: "age"
      value { float_list {
        value: 29.0
      }}
    }
    feature {
      key: "movie"
      value { bytes_list {
        value: "The Shawshank Redemption"
        value: "Fight Club"
      }}
    }
    feature {
      key: "movie_ratings"
      value { float_list {
        value: 9.0
        value: 9.7
      }}
    }
    feature {
      key: "suggestion"
      value { bytes_list {
        value: "Inception"
      }}
    }
    feature {
      key: "suggestion_purchased"
      value { int64_list {
        value: 1
      }}
    }
    feature {
      key: "purchase_price"
      value { float_list {
        value: 9.99
      }}
    }
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`feature`
</td>
<td>
`repeated FeatureEntry feature`
</td>
</tr>
</table>



## Child Classes
[`class FeatureEntry`](../../tf/train/Features/FeatureEntry.md)

