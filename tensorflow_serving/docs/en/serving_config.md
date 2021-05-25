# XGBoost Serving Configuration

In this guide, we will go over the configuration points for XGBoost Serving.

## Model Server Configuration

We only list some common configurations here, you can run `./tools/run_in_docker.sh ./bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --help` to get the full configurations.

### port

```
--port=8500                             int32   Port to listen on for gRPC API
```

### rest_api_port

```
--rest_api_port=0                       int32   Port to listen on for HTTP/REST API. If set to zero HTTP/REST API will not be exported. This port must be different than the one specified in --port.
```

### brpc_port

```
--brpc_port=0                           int32   Port to listen on for the BRPC dummy server
```

### platform_name

```
--platform_name="tensorflow"            string  Platform to use for serving:tensorflow/xgboost
```

### platform_config.txt

```
--platform_config_file=""               string  If non-empty, read an ascii PlatformConfigMap protobuf from the supplied file name, and use that platform config instead of the Tensorflow platform.
```

### model_config.txt

```
--model_config_file=""                  string  If non-empty, read an ascii ModelServerConfig protobuf from the supplied file name, and serve the models in that file. This config file can be used to specify multiple models to serve and other advanced parameters including non-default version policy.
```

The Model Server configuration file provided must be an ASCII [ModelServerConfig](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/config/model_server_config.proto#L76) protocol buffer. Refer to the following to understand [what an ASCII protocol buffer looks like](https://stackoverflow.com/questions/18873924/what-does-the-protobuf-text-format-look-like).

For all but the most advanced use-cases, you'll want to use the ModelConfigList option, which is a list of [ModelConfig](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/config/model_server_config.proto#L19) protocol buffers.

Here's a basic example, before we dive into advanced options below.

```
model_config_list {
  config {
    name: "test"
    base_path: "/tmp/test_model"
    model_platform: "xgboost"
    model_version_policy {
      latest {
        num_versions: 2
      }
    }
  }
}
```

Each ModelConfig specifies one model to be served, including its name and the absolute path where the Model Server should look for versions of the model to serve, as seen in the above example.

If you use the XGBoost model only, you should set the field model_platform to `xgboost` and if you use the XGBoost and regular FM model, you should set the field model_platform to `alphafm` and if you use the XGBoost and alphaFM_softmax model, you should set the field model_platform to `alphafm_softmax`.

By default the server will serve the version with the largest version number. This default can be overridden by changing the model_version_policy field. You can refer [Model Version Policy Configuration](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/config/file_system_storage_path_source.proto#L8) for all supported fields.
