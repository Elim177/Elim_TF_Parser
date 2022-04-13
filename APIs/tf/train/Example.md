description: An Example is a mostly-normalized data format for storing data for training and inference.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.train.Example" />
<meta itemprop="path" content="Stable" />
</div>

# tf.train.Example

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/core/example/example.proto">View source</a>



An `Example` is a mostly-normalized data format for storing data for training and inference.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.train.Example`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

It contains a key-value store `features` where each key (string) maps to a
<a href="../../tf/train/Feature.md"><code>tf.train.Feature</code></a> message. This flexible and compact format allows the
storage of large amounts of typed data, but requires that the data shape
and use be determined by the configuration files and parsers that are used to
read and write this format.

In TensorFlow, `Example`s are read in row-major
format, so any configuration that describes data with rank-2 or above
should keep this in mind. For example, to store an `M x N` matrix of bytes,
the <a href="../../tf/train/BytesList.md"><code>tf.train.BytesList</code></a> must contain M*N bytes, with `M` rows of `N` contiguous values
each. That is, the `BytesList` value must store the matrix as:

```.... row 0 .... // .... row 1 .... // ...........  // ... row M-1 ....```

An `Example` for a movie recommendation application:

```
    features {
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
      Note:that this feature exists to be used as a label in training.
      # E.g., if training a logistic regression model to predict purchase
      # probability in our learning tool we would set the label feature to
      # "suggestion_purchased".
      feature {
        key: "suggestion_purchased"
        value { float_list {
          value: 1.0
        }}
      }
      # Similar to "suggestion_purchased" above this feature exists to be used
      # as a label in training.
      # E.g., if training a linear regression model to predict purchase
      # price in our learning tool we would set the label feature to
      # "purchase_price".
      feature {
        key: "purchase_price"
        value { float_list {
          value: 9.99
        }}
      }
    }
```
A conformant `Example` dataset obeys the following conventions:

  - If a Feature `K` exists in one example with data type `T`, it must be of
      type `T` in all other examples when present. It may be omitted.
  - The number of instances of Feature `K` list data may vary across examples,
      depending on the requirements of the model.
  - If a Feature `K` doesn't exist in an example, a `K`-specific default will be
      used, if configured.
  - If a Feature `K` exists in an example but contains no items, the intent
      is considered to be an empty tensor and no default will be used.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`features`
</td>
<td>
`Features features`
</td>
</tr>
</table>



