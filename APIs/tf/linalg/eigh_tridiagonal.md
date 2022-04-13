description: Computes the eigenvalues of a Hermitian tridiagonal matrix.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.linalg.eigh_tridiagonal" />
<meta itemprop="path" content="Stable" />
</div>

# tf.linalg.eigh_tridiagonal

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/ops/linalg/linalg_impl.py">View source</a>



Computes the eigenvalues of a Hermitian tridiagonal matrix.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.linalg.eigh_tridiagonal`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.linalg.eigh_tridiagonal(
    alpha, beta, eigvals_only=(True), select=&#x27;a&#x27;, select_range=None,
    tol=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`alpha`
</td>
<td>
A real or complex tensor of shape (n), the diagonal elements of the
matrix. NOTE: If alpha is complex, the imaginary part is ignored (assumed
  zero) to satisfy the requirement that the matrix be Hermitian.
</td>
</tr><tr>
<td>
`beta`
</td>
<td>
A real or complex tensor of shape (n-1), containing the elements of
the first super-diagonal of the matrix. If beta is complex, the first
sub-diagonal of the matrix is assumed to be the conjugate of beta to
satisfy the requirement that the matrix be Hermitian
</td>
</tr><tr>
<td>
`eigvals_only`
</td>
<td>
If False, both eigenvalues and corresponding eigenvectors are
computed. If True, only eigenvalues are computed. Default is True.
</td>
</tr><tr>
<td>
`select`
</td>
<td>
Optional string with values in {‘a’, ‘v’, ‘i’} (default is 'a') that
determines which eigenvalues to calculate:
  'a': all eigenvalues.
  ‘v’: eigenvalues in the interval (min, max] given by `select_range`.
  'i’: eigenvalues with indices min <= i <= max.
</td>
</tr><tr>
<td>
`select_range`
</td>
<td>
Size 2 tuple or list or tensor specifying the range of
eigenvalues to compute together with select. If select is 'a',
select_range is ignored.
</td>
</tr><tr>
<td>
`tol`
</td>
<td>
Optional scalar. The absolute tolerance to which each eigenvalue is
required. An eigenvalue (or cluster) is considered to have converged if it
lies in an interval of this width. If tol is None (default), the value
eps*|T|_2 is used where eps is the machine precision, and |T|_2 is the
2-norm of the matrix T.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optional name of the op.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`eig_vals`
</td>
<td>
The eigenvalues of the matrix in non-decreasing order.
</td>
</tr><tr>
<td>
`eig_vectors`
</td>
<td>
If `eigvals_only` is False the eigenvectors are returned in
the second output argument.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If input values are invalid.
</td>
</tr><tr>
<td>
`NotImplemented`
</td>
<td>
Computing eigenvectors for `eigvals_only` = False is
not implemented yet.
</td>
</tr>
</table>


This op implements a subset of the functionality of
scipy.linalg.eigh_tridiagonal.

Note: The result is undefined if the input contains +/-inf or NaN, or if
any value in beta has a magnitude greater than
`numpy.sqrt(numpy.finfo(beta.dtype.as_numpy_dtype).max)`.



  Add support for outer batch dimensions.

#### Examples

```python
import numpy
eigvals = tf.linalg.eigh_tridiagonal([0.0, 0.0, 0.0], [1.0, 1.0])
eigvals_expected = [-numpy.sqrt(2.0), 0.0, numpy.sqrt(2.0)]
tf.assert_near(eigvals_expected, eigvals)
# ==> True
```