description: Generates skipgram word pairs.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.preprocessing.sequence.skipgrams" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.preprocessing.sequence.skipgrams

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras-preprocessing/tree/1.1.2/keras_preprocessing/sequence.py#L152-L240">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Generates skipgram word pairs.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.preprocessing.sequence.skipgrams`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.preprocessing.sequence.skipgrams(
    sequence, vocabulary_size, window_size=4, negative_samples=1.0, shuffle=(True),
    categorical=(False), sampling_table=None, seed=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This function transforms a sequence of word indexes (list of integers)
into tuples of words of the form:

- (word, word in the same window), with label 1 (positive samples).
- (word, random word from the vocabulary), with label 0 (negative samples).

Read more about Skipgram in this gnomic paper by Mikolov et al.:
[Efficient Estimation of Word Representations in
Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Arguments</h2></th></tr>

<tr>
<td>
`sequence`
</td>
<td>
A word sequence (sentence), encoded as a list
of word indices (integers). If using a `sampling_table`,
word indices are expected to match the rank
of the words in a reference dataset (e.g. 10 would encode
the 10-th most frequently occurring token).
Note that index 0 is expected to be a non-word and will be skipped.
</td>
</tr><tr>
<td>
`vocabulary_size`
</td>
<td>
Int, maximum possible word index + 1
</td>
</tr><tr>
<td>
`window_size`
</td>
<td>
Int, size of sampling windows (technically half-window).
The window of a word `w_i` will be
`[i - window_size, i + window_size+1]`.
</td>
</tr><tr>
<td>
`negative_samples`
</td>
<td>
Float >= 0. 0 for no negative (i.e. random) samples.
1 for same number as positive samples.
</td>
</tr><tr>
<td>
`shuffle`
</td>
<td>
Whether to shuffle the word couples before returning them.
</td>
</tr><tr>
<td>
`categorical`
</td>
<td>
bool. if False, labels will be
integers (eg. `[0, 1, 1 .. ]`),
if `True`, labels will be categorical, e.g.
`[[1,0],[0,1],[0,1] .. ]`.
</td>
</tr><tr>
<td>
`sampling_table`
</td>
<td>
1D array of size `vocabulary_size` where the entry i
encodes the probability to sample a word of rank i.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
Random seed.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
couples, labels: where `couples` are int pairs and
`labels` are either 0 or 1.
</td>
</tr>

</table>



#### Note:

By convention, index 0 in the vocabulary is
a non-word and will be skipped.
