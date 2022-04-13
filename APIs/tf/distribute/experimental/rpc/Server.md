description: A Server base class for accepting RPCs for registered tf.functions.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.distribute.experimental.rpc.Server" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="create"/>
<meta itemprop="property" content="register"/>
<meta itemprop="property" content="start"/>
</div>

# tf.distribute.experimental.rpc.Server

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="/code/stable/tensorflow/python/distribute/experimental/rpc/rpc_ops.py">View source</a>



A Server base class for accepting RPCs for registered tf.functions.

<!-- Placeholder for "Used in" -->

Functions can be registered on the server and are exposed via RPCs.

## Methods

<h3 id="create"><code>create</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/distribute/experimental/rpc/rpc_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>create(
    rpc_layer, address
)
</code></pre>

Create TF RPC server at given address.


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
Address where RPC server is hosted.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An instance of <a href="../../../../tf/distribute/experimental/rpc/Server.md"><code>tf.distribute.experimental.rpc.Server</code></a> class.
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
</td>
</tr>

</table>



#### Example usage:


```
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


<h3 id="register"><code>register</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/distribute/experimental/rpc/rpc_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>register(
    method_name: str,
    func: Union[def_function.Function, tf_function.ConcreteFunction]
)
</code></pre>

Method for registering tf.function on server.

Registered methods can be invoked remotely from clients.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`method_name`
</td>
<td>
Name of the tf.function. Clients use this method_name to make
RPCs.
</td>
</tr><tr>
<td>
`func`
</td>
<td>
A <a href="../../../../tf/function.md"><code>tf.function</code></a> or ConcreteFunction to register.
</td>
</tr>
</table>



<h3 id="start"><code>start</code></h3>

<a target="_blank" href="/code/stable/tensorflow/python/distribute/experimental/rpc/rpc_ops.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>start()
</code></pre>

Starts the RPC server on provided address.

Server listens for new requests from client, once it is started.



