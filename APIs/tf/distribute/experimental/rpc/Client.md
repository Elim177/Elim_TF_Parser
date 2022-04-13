description: Client class for invoking RPCs to the server.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.distribute.experimental.rpc.Client" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="call"/>
<meta itemprop="property" content="create"/>
</div>

# tf.distribute.experimental.rpc.Client

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/distribute/experimental/rpc/rpc_ops.py">View source</a>



Client class for invoking RPCs to the server.

<!-- Placeholder for "Used in" -->


## Methods

<h3 id="call"><code>call</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/distribute/experimental/rpc/rpc_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>call(
    method_name: str,
    args: Optional[Sequence[core_tf_types.Tensor]] = None,
    output_specs=None,
    timeout_in_ms=0
)
</code></pre>

Method for making RPC calls to remote server.

This invokes RPC to the server, executing the registered method_name
remotely.
Args:
  method_name: Remote registered method to invoke
  args: List of arguments for the registered method.
  output_specs: Output specs for the output from method.
     For example, if tf.function is: @tf.function(input_signature=[
       tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.int32) ])
      def multiply_fn(a, b): return tf.math.multiply(a, b)
    output_spec is: tf.TensorSpec((), tf.int32)  If you have access to TF
      Function, the output specs can be generated
   from tf.function by calling: output_specs =
     tf.nest.map_structure(tf.type_spec_from_value,
     tf_function.get_concrete_function().structured_outputs  If output_specs
     are not provided, flattened list of tensors will be returned in
     response.
  timeout_in_ms: Timeout for this call. If 0, default client timeout will be
    used.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An instance of `StatusOrResult` class with the following available
methods.
  * `is_ok()`:
      Returns True of RPC was successful.
  * `get_error()`:
      Returns TF error_code and error message for the RPC.
  * `get_value()`:
      Returns the returned value from remote TF function execution
      when RPC is successful.

Calling any of the above methods will block till RPC is completed and
result is available.
</td>
</tr>

</table>



<h3 id="create"><code>create</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/distribute/experimental/rpc/rpc_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>create(
    rpc_layer, address, name=&#x27;&#x27;, timeout_in_ms=0
)
</code></pre>

Create TF RPC client to connect to the given address.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`rpc_layer`
</td>
<td>
Communication layer between client and server. Only "grpc" rpc
layer is supported at the moment.
</td>
</tr><tr>
<td>
`address`
</td>
<td>
Address of the server to connect the RPC client to.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Name of the RPC Client. You can create multiple clients connecting
to same server and distinguish them using different names.
</td>
</tr><tr>
<td>
`timeout_in_ms`
</td>
<td>
The default timeout to use for outgoing RPCs from client. 0
indicates no timeout. Exceeding timeout during RPC will raise
DeadlineExceeded error.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An instance of <a href="../../../../tf/distribute/experimental/rpc/Client.md"><code>tf.distribute.experimental.rpc.Client</code></a> with the following
dynamically added methods for eagerly created clients:
  * `Registered methods` e.g. multiply(**args):
      If Client is created when executing eagerly, client will request the
      list of registered methods from server during client creation.
      The convenience methods for RPCs will be dynamically added to the
      created Client instance.

      For example, when a server has method "multiply" registered, the
      client object created in eager mode will have 'multiply' method
      available. Users can use client.multiply(..) to make RPC, instead of
      client.call("multiply", ...)

      These methods are not available when Client is created inside a
      tf.function.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>
<tr class="alt">
<td colspan="2">
A ValueError if rpc_layer other than "grpc" is used. Only GRPC
  is supported at the moment.
A DeadlineExceeded exception in eager mode if timeout exceeds while
  creating and listing client methods.
</td>
</tr>

</table>



#### Example usage:

```
>>> # Have server already started.
>>> import portpicker
>>> @tf.function(input_signature=[
...      tf.TensorSpec([], tf.int32),
...      tf.TensorSpec([], tf.int32)])
... def remote_fn(a, b):
...   return tf.add(a, b)
```

```
>>> port = portpicker.pick_unused_port()
>>> address = "localhost:{}".format(port)
>>> server = tf.distribute.experimental.rpc.Server.create("grpc", address)
>>> server.register("addition", remote_fn)
>>> server.start()
```

```
>>> # Start client
>>> client = tf.distribute.experimental.rpc.Client.create("grpc",
...      address=address, name="test_client")
```

```
>>> a = tf.constant(2, dtype=tf.int32)
>>> b = tf.constant(3, dtype=tf.int32)
```

```
>>> result = client.call(
...    args=[a, b],
...    method_name="addition",
...    output_specs=tf.TensorSpec((), tf.int32))
```

```
>>> if result.is_ok():
...   result.get_value()
```

```
>>> result = client.addition(a, b)
```

```
>>> if result.is_ok():
...   result.get_value()
```




