description: Public API for tf.keras.metrics namespace.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.keras.metrics" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.compat.v1.keras.metrics

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Public API for tf.keras.metrics namespace.



## Classes

[`class AUC`](../../../../tf/keras/metrics/AUC.md): Approximates the AUC (Area under the curve) of the ROC or PR curves.

[`class Accuracy`](../../../../tf/keras/metrics/Accuracy.md): Calculates how often predictions equal labels.

[`class BinaryAccuracy`](../../../../tf/keras/metrics/BinaryAccuracy.md): Calculates how often predictions match binary labels.

[`class BinaryCrossentropy`](../../../../tf/keras/metrics/BinaryCrossentropy.md): Computes the crossentropy metric between the labels and predictions.

[`class CategoricalAccuracy`](../../../../tf/keras/metrics/CategoricalAccuracy.md): Calculates how often predictions match one-hot labels.

[`class CategoricalCrossentropy`](../../../../tf/keras/metrics/CategoricalCrossentropy.md): Computes the crossentropy metric between the labels and predictions.

[`class CategoricalHinge`](../../../../tf/keras/metrics/CategoricalHinge.md): Computes the categorical hinge metric between `y_true` and `y_pred`.

[`class CosineSimilarity`](../../../../tf/keras/metrics/CosineSimilarity.md): Computes the cosine similarity between the labels and predictions.

[`class FalseNegatives`](../../../../tf/keras/metrics/FalseNegatives.md): Calculates the number of false negatives.

[`class FalsePositives`](../../../../tf/keras/metrics/FalsePositives.md): Calculates the number of false positives.

[`class Hinge`](../../../../tf/keras/metrics/Hinge.md): Computes the hinge metric between `y_true` and `y_pred`.

[`class KLDivergence`](../../../../tf/keras/metrics/KLDivergence.md): Computes Kullback-Leibler divergence metric between `y_true` and `y_pred`.

[`class LogCoshError`](../../../../tf/keras/metrics/LogCoshError.md): Computes the logarithm of the hyperbolic cosine of the prediction error.

[`class Mean`](../../../../tf/keras/metrics/Mean.md): Computes the (weighted) mean of the given values.

[`class MeanAbsoluteError`](../../../../tf/keras/metrics/MeanAbsoluteError.md): Computes the mean absolute error between the labels and predictions.

[`class MeanAbsolutePercentageError`](../../../../tf/keras/metrics/MeanAbsolutePercentageError.md): Computes the mean absolute percentage error between `y_true` and `y_pred`.

[`class MeanIoU`](../../../../tf/keras/metrics/MeanIoU.md): Computes the mean Intersection-Over-Union metric.

[`class MeanMetricWrapper`](../../../../tf/keras/metrics/MeanMetricWrapper.md): Wraps a stateless metric function with the Mean metric.

[`class MeanRelativeError`](../../../../tf/keras/metrics/MeanRelativeError.md): Computes the mean relative error by normalizing with the given values.

[`class MeanSquaredError`](../../../../tf/keras/metrics/MeanSquaredError.md): Computes the mean squared error between `y_true` and `y_pred`.

[`class MeanSquaredLogarithmicError`](../../../../tf/keras/metrics/MeanSquaredLogarithmicError.md): Computes the mean squared logarithmic error between `y_true` and `y_pred`.

[`class MeanTensor`](../../../../tf/keras/metrics/MeanTensor.md): Computes the element-wise (weighted) mean of the given tensors.

[`class Metric`](../../../../tf/keras/metrics/Metric.md): Encapsulates metric logic and state.

[`class Poisson`](../../../../tf/keras/metrics/Poisson.md): Computes the Poisson metric between `y_true` and `y_pred`.

[`class Precision`](../../../../tf/keras/metrics/Precision.md): Computes the precision of the predictions with respect to the labels.

[`class PrecisionAtRecall`](../../../../tf/keras/metrics/PrecisionAtRecall.md): Computes best precision where recall is >= specified value.

[`class Recall`](../../../../tf/keras/metrics/Recall.md): Computes the recall of the predictions with respect to the labels.

[`class RecallAtPrecision`](../../../../tf/keras/metrics/RecallAtPrecision.md): Computes best recall where precision is >= specified value.

[`class RootMeanSquaredError`](../../../../tf/keras/metrics/RootMeanSquaredError.md): Computes root mean squared error metric between `y_true` and `y_pred`.

[`class SensitivityAtSpecificity`](../../../../tf/keras/metrics/SensitivityAtSpecificity.md): Computes best sensitivity where specificity is >= specified value.

[`class SparseCategoricalAccuracy`](../../../../tf/keras/metrics/SparseCategoricalAccuracy.md): Calculates how often predictions match integer labels.

[`class SparseCategoricalCrossentropy`](../../../../tf/keras/metrics/SparseCategoricalCrossentropy.md): Computes the crossentropy metric between the labels and predictions.

[`class SparseTopKCategoricalAccuracy`](../../../../tf/keras/metrics/SparseTopKCategoricalAccuracy.md): Computes how often integer targets are in the top `K` predictions.

[`class SpecificityAtSensitivity`](../../../../tf/keras/metrics/SpecificityAtSensitivity.md): Computes best specificity where sensitivity is >= specified value.

[`class SquaredHinge`](../../../../tf/keras/metrics/SquaredHinge.md): Computes the squared hinge metric between `y_true` and `y_pred`.

[`class Sum`](../../../../tf/keras/metrics/Sum.md): Computes the (weighted) sum of the given values.

[`class TopKCategoricalAccuracy`](../../../../tf/keras/metrics/TopKCategoricalAccuracy.md): Computes how often targets are in the top `K` predictions.

[`class TrueNegatives`](../../../../tf/keras/metrics/TrueNegatives.md): Calculates the number of true negatives.

[`class TruePositives`](../../../../tf/keras/metrics/TruePositives.md): Calculates the number of true positives.

## Functions

[`KLD(...)`](../../../../tf/keras/metrics/kl_divergence.md): Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

[`MAE(...)`](../../../../tf/keras/metrics/mean_absolute_error.md): Computes the mean absolute error between labels and predictions.

[`MAPE(...)`](../../../../tf/keras/metrics/mean_absolute_percentage_error.md): Computes the mean absolute percentage error between `y_true` and `y_pred`.

[`MSE(...)`](../../../../tf/keras/metrics/mean_squared_error.md): Computes the mean squared error between labels and predictions.

[`MSLE(...)`](../../../../tf/keras/metrics/mean_squared_logarithmic_error.md): Computes the mean squared logarithmic error between `y_true` and `y_pred`.

[`binary_accuracy(...)`](../../../../tf/keras/metrics/binary_accuracy.md): Calculates how often predictions match binary labels.

[`binary_crossentropy(...)`](../../../../tf/keras/metrics/binary_crossentropy.md): Computes the binary crossentropy loss.

[`categorical_accuracy(...)`](../../../../tf/keras/metrics/categorical_accuracy.md): Calculates how often predictions match one-hot labels.

[`categorical_crossentropy(...)`](../../../../tf/keras/metrics/categorical_crossentropy.md): Computes the categorical crossentropy loss.

[`cosine(...)`](../../../../tf/keras/losses/cosine_similarity.md): Computes the cosine similarity between labels and predictions.

[`cosine_proximity(...)`](../../../../tf/keras/losses/cosine_similarity.md): Computes the cosine similarity between labels and predictions.

[`deserialize(...)`](../../../../tf/keras/metrics/deserialize.md): Deserializes a serialized metric class/function instance.

[`get(...)`](../../../../tf/keras/metrics/get.md): Retrieves a Keras metric as a `function`/`Metric` class instance.

[`hinge(...)`](../../../../tf/keras/metrics/hinge.md): Computes the hinge loss between `y_true` and `y_pred`.

[`kl_divergence(...)`](../../../../tf/keras/metrics/kl_divergence.md): Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

[`kld(...)`](../../../../tf/keras/metrics/kl_divergence.md): Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

[`kullback_leibler_divergence(...)`](../../../../tf/keras/metrics/kl_divergence.md): Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

[`log_cosh(...)`](../../../../tf/keras/losses/log_cosh.md): Logarithm of the hyperbolic cosine of the prediction error.

[`logcosh(...)`](../../../../tf/keras/losses/log_cosh.md): Logarithm of the hyperbolic cosine of the prediction error.

[`mae(...)`](../../../../tf/keras/metrics/mean_absolute_error.md): Computes the mean absolute error between labels and predictions.

[`mape(...)`](../../../../tf/keras/metrics/mean_absolute_percentage_error.md): Computes the mean absolute percentage error between `y_true` and `y_pred`.

[`mean_absolute_error(...)`](../../../../tf/keras/metrics/mean_absolute_error.md): Computes the mean absolute error between labels and predictions.

[`mean_absolute_percentage_error(...)`](../../../../tf/keras/metrics/mean_absolute_percentage_error.md): Computes the mean absolute percentage error between `y_true` and `y_pred`.

[`mean_squared_error(...)`](../../../../tf/keras/metrics/mean_squared_error.md): Computes the mean squared error between labels and predictions.

[`mean_squared_logarithmic_error(...)`](../../../../tf/keras/metrics/mean_squared_logarithmic_error.md): Computes the mean squared logarithmic error between `y_true` and `y_pred`.

[`mse(...)`](../../../../tf/keras/metrics/mean_squared_error.md): Computes the mean squared error between labels and predictions.

[`msle(...)`](../../../../tf/keras/metrics/mean_squared_logarithmic_error.md): Computes the mean squared logarithmic error between `y_true` and `y_pred`.

[`poisson(...)`](../../../../tf/keras/metrics/poisson.md): Computes the Poisson loss between y_true and y_pred.

[`serialize(...)`](../../../../tf/keras/metrics/serialize.md): Serializes metric function or `Metric` instance.

[`sparse_categorical_accuracy(...)`](../../../../tf/keras/metrics/sparse_categorical_accuracy.md): Calculates how often predictions match integer labels.

[`sparse_categorical_crossentropy(...)`](../../../../tf/keras/metrics/sparse_categorical_crossentropy.md): Computes the sparse categorical crossentropy loss.

[`sparse_top_k_categorical_accuracy(...)`](../../../../tf/keras/metrics/sparse_top_k_categorical_accuracy.md): Computes how often integer targets are in the top `K` predictions.

[`squared_hinge(...)`](../../../../tf/keras/metrics/squared_hinge.md): Computes the squared hinge loss between `y_true` and `y_pred`.

[`top_k_categorical_accuracy(...)`](../../../../tf/keras/metrics/top_k_categorical_accuracy.md): Computes how often targets are in the top `K` predictions.

