description: Text tokenization utility class.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.preprocessing.text.Tokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="fit_on_sequences"/>
<meta itemprop="property" content="fit_on_texts"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="sequences_to_matrix"/>
<meta itemprop="property" content="sequences_to_texts"/>
<meta itemprop="property" content="sequences_to_texts_generator"/>
<meta itemprop="property" content="texts_to_matrix"/>
<meta itemprop="property" content="texts_to_sequences"/>
<meta itemprop="property" content="texts_to_sequences_generator"/>
<meta itemprop="property" content="to_json"/>
</div>

# tf.keras.preprocessing.text.Tokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras-preprocessing/tree/1.1.2/keras_preprocessing/text.py#L141-L487">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Text tokenization utility class.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.preprocessing.text.Tokenizer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.preprocessing.text.Tokenizer(
    num_words=None,
    filters=&#x27;!&quot;#$%&amp;()*+,-./:;&lt;=&gt;?@[\\]^_`{|}~\t\n&#x27;,
    lower=(True), split=&#x27; &#x27;, char_level=(False), oov_token=None,
    document_count=0, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This class allows to vectorize a text corpus, by turning each
text into either a sequence of integers (each integer being the index
of a token in a dictionary) or into a vector where the coefficient
for each token could be binary, based on word count, based on tf-idf...

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Arguments</h2></th></tr>

<tr>
<td>
`num_words`
</td>
<td>
the maximum number of words to keep, based
on word frequency. Only the most common `num_words-1` words will
be kept.
</td>
</tr><tr>
<td>
`filters`
</td>
<td>
a string where each element is a character that will be
filtered from the texts. The default is all punctuation, plus
tabs and line breaks, minus the `'` character.
</td>
</tr><tr>
<td>
`lower`
</td>
<td>
boolean. Whether to convert the texts to lowercase.
</td>
</tr><tr>
<td>
`split`
</td>
<td>
str. Separator for word splitting.
</td>
</tr><tr>
<td>
`char_level`
</td>
<td>
if True, every character will be treated as a token.
</td>
</tr><tr>
<td>
`oov_token`
</td>
<td>
if given, it will be added to word_index and used to
replace out-of-vocabulary words during text_to_sequence calls
</td>
</tr>
</table>


By default, all punctuation is removed, turning the texts into
space-separated sequences of words
(words maybe include the `'` character). These sequences are then
split into lists of tokens. They will then be indexed or vectorized.

`0` is a reserved index that won't be assigned to any word.

## Methods

<h3 id="fit_on_sequences"><code>fit_on_sequences</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras-preprocessing/tree/1.1.2/keras_preprocessing/text.py#L253-L267">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>fit_on_sequences(
    sequences
)
</code></pre>

Updates internal vocabulary based on a list of sequences.

Required before using `sequences_to_matrix`
(if `fit_on_texts` was never called).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`sequences`
</td>
<td>
A list of sequence.
A "sequence" is a list of integer word indices.
</td>
</tr>
</table>



<h3 id="fit_on_texts"><code>fit_on_texts</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras-preprocessing/tree/1.1.2/keras_preprocessing/text.py#L199-L251">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>fit_on_texts(
    texts
)
</code></pre>

Updates internal vocabulary based on a list of texts.

In the case where texts contains lists,
we assume each entry of the lists to be a token.

Required before using `texts_to_sequences` or `texts_to_matrix`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`texts`
</td>
<td>
can be a list of strings,
a generator of strings (for memory-efficiency),
or a list of list of strings.
</td>
</tr>
</table>



<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras-preprocessing/tree/1.1.2/keras_preprocessing/text.py#L440-L468">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

Returns the tokenizer configuration as Python dictionary.
The word count dictionaries used by the tokenizer get serialized
into plain JSON, so that the configuration can be read by other
projects.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Python dictionary with the tokenizer configuration.
</td>
</tr>

</table>



<h3 id="sequences_to_matrix"><code>sequences_to_matrix</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras-preprocessing/tree/1.1.2/keras_preprocessing/text.py#L385-L438">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>sequences_to_matrix(
    sequences, mode=&#x27;binary&#x27;
)
</code></pre>

Converts a list of sequences into a Numpy matrix.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`sequences`
</td>
<td>
list of sequences
(a sequence is a list of integer word indices).
</td>
</tr><tr>
<td>
`mode`
</td>
<td>
one of "binary", "count", "tfidf", "freq"
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Numpy matrix.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
In case of invalid `mode` argument,
or if the Tokenizer requires to be fit to sample data.
</td>
</tr>
</table>



<h3 id="sequences_to_texts"><code>sequences_to_texts</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras-preprocessing/tree/1.1.2/keras_preprocessing/text.py#L326-L338">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>sequences_to_texts(
    sequences
)
</code></pre>

Transforms each sequence into a list of text.

Only top `num_words-1` most frequent words will be taken into account.
Only words known by the tokenizer will be taken into account.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`sequences`
</td>
<td>
A list of sequences (list of integers).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of texts (strings)
</td>
</tr>

</table>



<h3 id="sequences_to_texts_generator"><code>sequences_to_texts_generator</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras-preprocessing/tree/1.1.2/keras_preprocessing/text.py#L340-L370">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>sequences_to_texts_generator(
    sequences
)
</code></pre>

Transforms each sequence in `sequences` to a list of texts(strings).

Each sequence has to a list of integers.
In other words, sequences should be a list of sequences

Only top `num_words-1` most frequent words will be taken into account.
Only words known by the tokenizer will be taken into account.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`sequences`
</td>
<td>
A list of sequences.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Yields</th></tr>
<tr class="alt">
<td colspan="2">
Yields individual texts.
</td>
</tr>

</table>



<h3 id="texts_to_matrix"><code>texts_to_matrix</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras-preprocessing/tree/1.1.2/keras_preprocessing/text.py#L372-L383">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>texts_to_matrix(
    texts, mode=&#x27;binary&#x27;
)
</code></pre>

Convert a list of texts to a Numpy matrix.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`texts`
</td>
<td>
list of strings.
</td>
</tr><tr>
<td>
`mode`
</td>
<td>
one of "binary", "count", "tfidf", "freq".
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Numpy matrix.
</td>
</tr>

</table>



<h3 id="texts_to_sequences"><code>texts_to_sequences</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras-preprocessing/tree/1.1.2/keras_preprocessing/text.py#L269-L281">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>texts_to_sequences(
    texts
)
</code></pre>

Transforms each text in texts to a sequence of integers.

Only top `num_words-1` most frequent words will be taken into account.
Only words known by the tokenizer will be taken into account.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`texts`
</td>
<td>
A list of texts (strings).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of sequences.
</td>
</tr>

</table>



<h3 id="texts_to_sequences_generator"><code>texts_to_sequences_generator</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras-preprocessing/tree/1.1.2/keras_preprocessing/text.py#L283-L324">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>texts_to_sequences_generator(
    texts
)
</code></pre>

Transforms each text in `texts` to a sequence of integers.

Each item in texts can also be a list,
in which case we assume each item of that list to be a token.

Only top `num_words-1` most frequent words will be taken into account.
Only words known by the tokenizer will be taken into account.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`texts`
</td>
<td>
A list of texts (strings).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Yields</th></tr>
<tr class="alt">
<td colspan="2">
Yields individual sequences.
</td>
</tr>

</table>



<h3 id="to_json"><code>to_json</code></h3>

<a target="_blank" href="https://github.com/keras-team/keras-preprocessing/tree/1.1.2/keras_preprocessing/text.py#L470-L487">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_json(
    **kwargs
)
</code></pre>

Returns a JSON string containing the tokenizer configuration.
To load a tokenizer from a JSON string, use
<a href="../../../../tf/keras/preprocessing/text/tokenizer_from_json.md"><code>keras.preprocessing.text.tokenizer_from_json(json_string)</code></a>.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`**kwargs`
</td>
<td>
Additional keyword arguments
to be passed to `json.dumps()`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A JSON string containing the tokenizer configuration.
</td>
</tr>

</table>





