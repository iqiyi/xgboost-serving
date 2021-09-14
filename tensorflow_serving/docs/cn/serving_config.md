# XGBoost Serving 配置

本文档介绍 XGBoost Serving 支持的主要配置。

## Model Server 配置

这里仅介绍常用的配置，运行 `./tools/run_in_docker.sh ./bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --help` 查看完整的配置。

### port

  ```
--port=8500                             int32   gRPC API 端口
  ```

### rest_api_port

  ```
--rest_api_port=0                       int32   HTTP/REST API 端口，该端口必须和 gRPC API 端口不同。
  ```

### brpc_port

  ```
--brpc_port=0                           int32   BRPC dummy server 端口
  ```

### platform_name

  ```
--platform_name="tensorflow"            string  使用的平台
  ```

### platform_config.txt

  ```
--platform_config_file=""               string  平台的配置
  ```

### model_config.txt

  ```
--model_config_file=""                  string  模型的配置，规范为 ModelServerConfig protobuf message。
  ```

ModelServerConfig protobuf message 请参考：[ModelServerConfig](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/config/model_server_config.proto#L76)。 ASCII protocol buffer 请参考：[what an ASCII protocol buffer looks like](https://stackoverflow.com/questions/18873924/what-does-the-protobuf-text-format-look-like)。

对大多数使用场景，请使用 ModelConfigList 配置，详细含义请参考：[ModelConfig](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/config/model_server_config.proto#L19)。

1 个简单的 ModelConfigList 配置如下：

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

每个 ModelConfig 配置 1 个加载的模型，包括模型的名字和绝对路径。

如果仅使用 XGBoost 模型，设置 model_platform 为 `xgboost`，如果使用 XGBoost 和 FM 模型，设置 model_platform 为 `alphafm`，如果使用 XGBoost 和 alphaFM_softmax 模型，设置 model_platform 为 `alphafm_softmax`。

缺省配置下，model server 会加载最新版本的模型。用户可以通过修改 model_version_policy 部分来覆盖缺省配置，详细请参考：[Model Version Policy Configuration](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/config/file_system_storage_path_source.proto#L8)。
