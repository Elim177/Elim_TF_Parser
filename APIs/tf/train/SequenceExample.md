description: A SequenceExample is a format for representing one or more sequences and some context.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.train.SequenceExample" />
<meta itemprop="path" content="Stable" />
</div>

# tf.train.SequenceExample

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/core/example/example.proto">View source</a>



A `SequenceExample` is a format for representing one or more sequences and some context.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.train.SequenceExample`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

The `context` contains features which apply to the entire
example. The `feature_lists` contain a key, value map where each key is
associated with a repeated set of <a href="../../tf/train/Features.md"><code>tf.train.Features</code></a> (a <a href="../../tf/train/FeatureList.md"><code>tf.train.FeatureList</code></a>).
A `FeatureList` represents the values of a feature identified by its key
over time / frames.

Below is a `SequenceExample` for a movie recommendation application recording a
sequence of ratings by a user. The time-independent features ("locale",
"age", "favorites") describing the user are part of the context. The sequence
of movies the user rated are part of the feature_lists. For each movie in the
sequence we have information on its name and actors and the user's rating.
This information is recorded in three separate `feature_list`s.
In the example below there are only two movies. All three `feature_list`s,
namely "movie_ratings", "movie_names", and "actors" have a feature value for
both movies. Note, that "actors" is itself a `bytes_list` with multiple
strings per movie.

```
  context: {
    feature: {
      key  : "locale"
      value: {
        bytes_list: {
          value: [ "pt_BR" ]
        }
      }
    }
    feature: {
      key  : "age"
      value: {
        float_list: {
          value: [ 19.0 ]
        }
      }
    }
    feature: {
      key  : "favorites"
      value: {
        bytes_list: {
          value: [ "Majesty Rose", "Savannah Outen", "One Direction" ]
        }
      }
    }
  }
  feature_lists: {
    feature_list: {
      key  : "movie_ratings"
      value: {
        feature: {
          float_list: {
            value: [ 4.5 ]
          }
        }
        feature: {
          float_list: {
            value: [ 5.0 ]
          }
        }
      }
    }
    feature_list: {
      key  : "movie_names"
      value: {
        feature: {
          bytes_list: {
            value: [ "The Shawshank Redemption" ]
          }
        }
        feature: {
          bytes_list: {
            value: [ "Fight Club" ]
          }
        }
      }
    }
    feature_list: {
      key  : "actors"
      value: {
        feature: {
          bytes_list: {
            value: [ "Tim Robbins", "Morgan Freeman" ]
          }
        }
        feature: {
          bytes_list: {
            value: [ "Brad Pitt", "Edward Norton", "Helena Bonham Carter" ]
          }
        }
      }
    }
  }
```

A conformant `SequenceExample` data set obeys the following conventions:

`context`:

  - All conformant context features `K` must obey the same conventions as
    a conformant Example's features (see above).

`feature_lists`:

  - A `FeatureList L` may be missing in an example; it is up to the
    parser configuration to determine if this is allowed or considered
    an empty list (zero length).
  - If a `FeatureList L` exists, it may be empty (zero length).
  - If a `FeatureList L` is non-empty, all features within the `FeatureList`
    must have the same data type `T`. Even across `SequenceExample`s, the type `T`
    of the `FeatureList` identified by the same key must be the same. An entry
    without any values may serve as an empty feature.
  - If a `FeatureList L` is non-empty, it is up to the parser configuration
    to determine if all features within the `FeatureList` must
    have the same size.  The same holds for this `FeatureList` across multiple
    examples.
  - For sequence modeling ([example](https://github.com/tensorflow/nmt)), the
    feature lists represent a sequence of frames. In this scenario, all
    `FeatureList`s in a `SequenceExample` have the same number of `Feature`
    messages, so that the i-th element in each `FeatureList` is part of the
    i-th frame (or time step).

**Examples of conformant and non-conformant examples' `FeatureLists`:**

Conformant `FeatureLists`:

```
    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.5 ] } }
               feature: { float_list: { value: [ 5.0 ] } } }
    } }
```

Non-conformant `FeatureLists` (mismatched types):

```
    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.5 ] } }
               feature: { int64_list: { value: [ 5 ] } } }
    } }
```

Conditionally conformant `FeatureLists`, the parser configuration determines
if the feature sizes must match:

```
    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.5 ] } }
               feature: { float_list: { value: [ 5.0, 6.0 ] } } }
    } }
```

**Examples of conformant and non-conformant `SequenceExample`s:**

Conformant pair of SequenceExample:

```
    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.5 ] } }
               feature: { float_list: { value: [ 5.0 ] } } }
     } }

    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.5 ] } }
               feature: { float_list: { value: [ 5.0 ] } }
               feature: { float_list: { value: [ 2.0 ] } } }
     } }
```

Conformant pair of `SequenceExample`s:

```
    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.5 ] } }
               feature: { float_list: { value: [ 5.0 ] } } }
     } }

    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { }
     } }
```

Conditionally conformant pair of `SequenceExample`s, the parser configuration
determines if the second `feature_lists` is consistent (zero-length) or
invalid (missing "movie_ratings"):

```
    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.5 ] } }
               feature: { float_list: { value: [ 5.0 ] } } }
     } }

   feature_lists: { }
```

Non-conformant pair of `SequenceExample`s (mismatched types):

```
    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.5 ] } }
               feature: { float_list: { value: [ 5.0 ] } } }
     } }

    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { int64_list: { value: [ 4 ] } }
               feature: { int64_list: { value: [ 5 ] } }
               feature: { int64_list: { value: [ 2 ] } } }
     } }
```

Conditionally conformant pair of `SequenceExample`s; the parser configuration
determines if the feature sizes must match:

```
    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.5 ] } }
               feature: { float_list: { value: [ 5.0 ] } } }
    } }

    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.0 ] } }
              feature: { float_list: { value: [ 5.0, 3.0 ] } }
    } }
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`context`
</td>
<td>
`Features context`
</td>
</tr><tr>
<td>
`feature_lists`
</td>
<td>
`FeatureLists feature_lists`
</td>
</tr>
</table>



