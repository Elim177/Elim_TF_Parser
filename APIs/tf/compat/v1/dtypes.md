description: Public API for tf.dtypes namespace.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.dtypes" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="QUANTIZED_DTYPES"/>
<meta itemprop="property" content="bfloat16"/>
<meta itemprop="property" content="bool"/>
<meta itemprop="property" content="complex128"/>
<meta itemprop="property" content="complex64"/>
<meta itemprop="property" content="double"/>
<meta itemprop="property" content="float16"/>
<meta itemprop="property" content="float32"/>
<meta itemprop="property" content="float64"/>
<meta itemprop="property" content="half"/>
<meta itemprop="property" content="int16"/>
<meta itemprop="property" content="int32"/>
<meta itemprop="property" content="int64"/>
<meta itemprop="property" content="int8"/>
<meta itemprop="property" content="qint16"/>
<meta itemprop="property" content="qint32"/>
<meta itemprop="property" content="qint8"/>
<meta itemprop="property" content="quint16"/>
<meta itemprop="property" content="quint8"/>
<meta itemprop="property" content="resource"/>
<meta itemprop="property" content="string"/>
<meta itemprop="property" content="uint16"/>
<meta itemprop="property" content="uint32"/>
<meta itemprop="property" content="uint64"/>
<meta itemprop="property" content="uint8"/>
<meta itemprop="property" content="variant"/>
</div>

# Module: tf.compat.v1.dtypes

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Public API for tf.dtypes namespace.



## Classes

[`class DType`](../../../tf/dtypes/DType.md): Represents the type of the elements in a `Tensor`.

## Functions

[`as_dtype(...)`](../../../tf/dtypes/as_dtype.md): Converts the given `type_value` to a `DType`.

[`as_string(...)`](../../../tf/strings/as_string.md): Converts each entry in the given tensor to strings.

[`cast(...)`](../../../tf/cast.md): Casts a tensor to a new type.

[`complex(...)`](../../../tf/dtypes/complex.md): Converts two real numbers to a complex number.

[`saturate_cast(...)`](../../../tf/dtypes/saturate_cast.md): Performs a safe saturating cast of `value` to `dtype`.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
QUANTIZED_DTYPES<a id="QUANTIZED_DTYPES"></a>
</td>
<td>
```
{
 tf.qint16,
 tf.qint16_ref,
 tf.qint32,
 tf.qint32_ref,
 tf.qint8,
 tf.qint8_ref,
 tf.quint16,
 tf.quint16_ref,
 tf.quint8,
 tf.quint8_ref
}
```
</td>
</tr><tr>
<td>
bfloat16<a id="bfloat16"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

16-bit bfloat (brain floating point).
</td>
</tr><tr>
<td>
bool<a id="bool"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

Boolean.
</td>
</tr><tr>
<td>
complex128<a id="complex128"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

128-bit complex.
</td>
</tr><tr>
<td>
complex64<a id="complex64"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

64-bit complex.
</td>
</tr><tr>
<td>
double<a id="double"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

64-bit (double precision) floating-point.
</td>
</tr><tr>
<td>
float16<a id="float16"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

16-bit (half precision) floating-point.
</td>
</tr><tr>
<td>
float32<a id="float32"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

32-bit (single precision) floating-point.
</td>
</tr><tr>
<td>
float64<a id="float64"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

64-bit (double precision) floating-point.
</td>
</tr><tr>
<td>
half<a id="half"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

16-bit (half precision) floating-point.
</td>
</tr><tr>
<td>
int16<a id="int16"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

Signed 16-bit integer.
</td>
</tr><tr>
<td>
int32<a id="int32"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

Signed 32-bit integer.
</td>
</tr><tr>
<td>
int64<a id="int64"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

Signed 64-bit integer.
</td>
</tr><tr>
<td>
int8<a id="int8"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

Signed 8-bit integer.
</td>
</tr><tr>
<td>
qint16<a id="qint16"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

Signed quantized 16-bit integer.
</td>
</tr><tr>
<td>
qint32<a id="qint32"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

signed quantized 32-bit integer.
</td>
</tr><tr>
<td>
qint8<a id="qint8"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

Signed quantized 8-bit integer.
</td>
</tr><tr>
<td>
quint16<a id="quint16"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

Unsigned quantized 16-bit integer.
</td>
</tr><tr>
<td>
quint8<a id="quint8"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

Unsigned quantized 8-bit integer.
</td>
</tr><tr>
<td>
resource<a id="resource"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

Handle to a mutable, dynamically allocated resource.
</td>
</tr><tr>
<td>
string<a id="string"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

Variable-length string, represented as byte array.
</td>
</tr><tr>
<td>
uint16<a id="uint16"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

Unsigned 32-bit (dword) integer.
</td>
</tr><tr>
<td>
uint32<a id="uint32"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>
</td>
</tr><tr>
<td>
uint64<a id="uint64"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

Unsigned 64-bit (qword) integer.
</td>
</tr><tr>
<td>
uint8<a id="uint8"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

Unsigned 8-bit (byte) integer.
</td>
</tr><tr>
<td>
variant<a id="variant"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>

Data of arbitrary type (known at runtime).
</td>
</tr>
</table>

