description: Configuration class for tf.data service dispatchers.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.service.DispatcherConfig" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="fault_tolerant_mode"/>
<meta itemprop="property" content="job_gc_check_interval_ms"/>
<meta itemprop="property" content="job_gc_timeout_ms"/>
<meta itemprop="property" content="port"/>
<meta itemprop="property" content="protocol"/>
<meta itemprop="property" content="work_dir"/>
<meta itemprop="property" content="worker_addresses"/>
</div>

# tf.data.experimental.service.DispatcherConfig

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/data/experimental/service/server_lib.py">View source</a>



Configuration class for tf.data service dispatchers.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.data.experimental.service.DispatcherConfig`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.service.DispatcherConfig(
    port=0, protocol=None, work_dir=None, fault_tolerant_mode=(False),
    worker_addresses=None, job_gc_check_interval_ms=None, job_gc_timeout_ms=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Fields:


* <b>`port`</b>: Specifies the port to bind to. A value of 0 indicates that the server
  may bind to any available port.
* <b>`protocol`</b>: The protocol to use for communicating with the tf.data service,
  e.g. "grpc".
* <b>`work_dir`</b>: A directory to store dispatcher state in. This
  argument is required for the dispatcher to be able to recover from
  restarts.
* <b>`fault_tolerant_mode`</b>: Whether the dispatcher should write its state to a
  journal so that it can recover from restarts. Dispatcher state, including
  registered datasets and created jobs, is synchronously written to the
  journal before responding to RPCs. If `True`, `work_dir` must also be
  specified.
* <b>`worker_addresses`</b>: If the job uses auto-sharding, it needs to specify a fixed
  list of worker addresses that will register with the dispatcher. The
  worker addresses should be in the format `"host"` or `"host:port"`, where
  `"port"` is an integer, named port, or `%port%` to match any port.
* <b>`job_gc_check_interval_ms`</b>: How often the dispatcher should scan through to
  delete old and unused jobs, in milliseconds. If not set, the runtime will
  select a reasonable default. A higher value will reduce load on the
  dispatcher, while a lower value will reduce the time it takes for the
  dispatcher to garbage collect expired jobs.
* <b>`job_gc_timeout_ms`</b>: How long a job needs to be unused before it becomes a
  candidate for garbage collection, in milliseconds. A value of -1 indicates
  that jobs should never be garbage collected. If not set, the runtime will
  select a reasonable default. A higher value will cause jobs to stay around
  longer with no consumers. This is useful if there is a large gap in
  time between when consumers read from the job. A lower value will reduce
  the time it takes to reclaim the resources from expired jobs.




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
fault_tolerant_mode<a id="fault_tolerant_mode"></a>
</td>
<td>
Instance of `_collections._tuplegetter`

Alias for field number 3
</td>
</tr><tr>
<td>
job_gc_check_interval_ms<a id="job_gc_check_interval_ms"></a>
</td>
<td>
Instance of `_collections._tuplegetter`

Alias for field number 5
</td>
</tr><tr>
<td>
job_gc_timeout_ms<a id="job_gc_timeout_ms"></a>
</td>
<td>
Instance of `_collections._tuplegetter`

Alias for field number 6
</td>
</tr><tr>
<td>
port<a id="port"></a>
</td>
<td>
Instance of `_collections._tuplegetter`

Alias for field number 0
</td>
</tr><tr>
<td>
protocol<a id="protocol"></a>
</td>
<td>
Instance of `_collections._tuplegetter`

Alias for field number 1
</td>
</tr><tr>
<td>
work_dir<a id="work_dir"></a>
</td>
<td>
Instance of `_collections._tuplegetter`

Alias for field number 2
</td>
</tr><tr>
<td>
worker_addresses<a id="worker_addresses"></a>
</td>
<td>
Instance of `_collections._tuplegetter`

Alias for field number 4
</td>
</tr>
</table>

