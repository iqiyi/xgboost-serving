# 部署 XGBoost && FM 模型

本文档介绍如何使用 XGBoost Serving 部署 XGBoost && FM 模型，如果想了解如何训练并导出 XGBoost && FM 模型，请参考：[导出 XGBoost && FM 模型](export_model.md)。

## 使用 XGBoost Serving model server 加载模型

如果仅使用 XGBoost 模型，运行以下命令加载模型：

  ```
./tools/run_in_docker.sh ./bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --platform_name=xgboost --platform_config_file=./tensorflow_serving/servables/xgboost/testdata/platform_config.txt --model_config_file=./tensorflow_serving/servables/xgboost/testdata/model_config.txt --port=8500 --rest_api_port=8501 --brpc_port=9999
  ```

**注意: 请修改 `model_config.txt` 配置文件中的 `base_path` 参数为待加载模型的绝对路径。**

如果 model server 成功加载模型，则会显示以下日志：

  ```
2021-05-20 09:55:53.979454: I tensorflow_serving/model_servers/server_core.cc:460] Adding/updating models.
2021-05-20 09:55:53.979491: I tensorflow_serving/model_servers/server_core.cc:571]  (Re-)adding model: test
2021-05-20 09:55:54.079710: I tensorflow_serving/core/basic_manager.cc:739] Successfully reserved resources to load servable {name: test version: 1}
2021-05-20 09:55:54.079747: I tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: test version: 1}
2021-05-20 09:55:54.079755: I tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: test version: 1}
2021-05-20 09:55:54.079763: I tensorflow_serving/servables/xgboost/xgboost_bundle.cc:26] Call the constructor XgboostBundle().
[09:55:54] WARNING: /tmp/xgboost.f9ak1c/src/objective/regression_obj.cu:171: reg:linear is now deprecated in favor of reg:squarederror.
[09:55:54] WARNING: /tmp/xgboost.f9ak1c/src/learner.cc:851: Loading model from XGBoost < 1.0.0, consider saving it again for improved compatibility
2021-05-20 09:55:54.082886: I tensorflow_serving/core/loader_harness.cc:87] Successfully loaded servable version {name: test version: 1}
I0520 09:55:54.083896    27 tensorflow_serving/model_servers/server.cc:279] Running gRPC ModelServer at 0.0.0.0:8500 ...
[evhttp_server.cc : 238] NET_LOG: Entering the event loop ...
I0520 09:55:54.084813    27 tensorflow_serving/model_servers/server.cc:299] Exporting HTTP/REST API at:localhost:8501 ...
I0520 09:55:54.088261    27 external/brpc/src/brpc/server.cpp:1045] Server[DummyServerOf(./bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server)] is serving on port=9999.
I0520 09:55:54.088595    27 external/brpc/src/brpc/server.cpp:1048] Check out http://haoziyu-workstation:9999 in web browser.
  ```

如果使用 XGBoost 和 FM 模型，运行以下命令加载模型：

```
./tools/run_in_docker.sh ./bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --platform_name=xgboost --platform_config_file=./tensorflow_serving/servables/alphafm/testdata/platform_config.txt --model_config_file=./tensorflow_serving/servables/alphafm/testdata/model_config.txt --port=8500 --rest_api_port=8501 --brpc_port=9999
```

**注意: 请修改 `model_config.txt` 配置文件中的 `base_path` 参数为待加载模型的绝对路径。**

如果 model server 成功加载模型，则会显示以下日志：

  ```
2021-05-20 10:00:16.385993: I tensorflow_serving/model_servers/server_core.cc:460] Adding/updating models.
2021-05-20 10:00:16.386019: I tensorflow_serving/model_servers/server_core.cc:571]  (Re-)adding model: test
2021-05-20 10:00:16.486242: I tensorflow_serving/core/basic_manager.cc:739] Successfully reserved resources to load servable {name: test version: 1}
2021-05-20 10:00:16.486279: I tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: test version: 1}
2021-05-20 10:00:16.486287: I tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: test version: 1}
2021-05-20 10:00:16.486295: I tensorflow_serving/servables/alphafm/alphafm_bundle.cc:26] Call the constructor AlphafmBundle().
[10:00:16] WARNING: /tmp/xgboost.f9ak1c/src/objective/regression_obj.cu:171: reg:linear is now deprecated in favor of reg:squarederror.
[10:00:16] WARNING: /tmp/xgboost.f9ak1c/src/learner.cc:851: Loading model from XGBoost < 1.0.0, consider saving it again for improved compatibility
2021-05-20 10:00:16.486779: I tensorflow_serving/servables/alphafm/alphafm_model.cc:33] Call the constructor AlphaFmModel().
2021-05-20 10:00:16.486787: I tensorflow_serving/servables/alphafm/alphafm_model.cc:52] Call the method LoadModel(model_path).
2021-05-20 10:00:16.526347: W tensorflow_serving/servables/alphafm/alphafm_model.cc:77] Empty line
2021-05-20 10:00:16.526406: I tensorflow_serving/servables/alphafm/feature_mapping.cc:28] Call the constructor FeatureMapping().
2021-05-20 10:00:16.526421: I tensorflow_serving/servables/alphafm/feature_mapping.cc:47] Call the method LoadModel(model_path).
2021-05-20 10:00:16.528165: W tensorflow_serving/servables/alphafm/feature_mapping.cc:57] Empty line
2021-05-20 10:00:16.528177: I tensorflow_serving/core/loader_harness.cc:87] Successfully loaded servable version {name: test version: 1}
I0520 10:00:16.529194    27 tensorflow_serving/model_servers/server.cc:279] Running gRPC ModelServer at 0.0.0.0:8500 ...
[evhttp_server.cc : 238] NET_LOG: Entering the event loop ...
I0520 10:00:16.529838    27 tensorflow_serving/model_servers/server.cc:299] Exporting HTTP/REST API at:localhost:8501 ...
I0520 10:00:16.531818    27 external/brpc/src/brpc/server.cpp:1045] Server[DummyServerOf(./bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server)] is serving on port=9999.
I0520 10:00:16.532032    27 external/brpc/src/brpc/server.cpp:1048] Check out http://haoziyu-workstation:9999 in web browser.
  ```

如果使用 XGBoost 和 alphaFM_softmax 模型，运行以下命令加载模型：

  ```
./bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --platform_name=xgboost --platform_config_file=./tensorflow_serving/servables/alphafm_softmax/testdata/platform_config.txt --model_config_file=./tensorflow_serving/servables/alphafm_softmax/testdata/model_config.txt --port=8500 --rest_api_port=8501 --brpc_port=9999
  ```

**注意: 请修改 `model_config.txt` 配置文件中的 `base_path` 参数为待加载模型的绝对路径。**

如果 model server 成功加载模型，则会显示以下日志：

  ```
2021-05-21 10:55:08.579463: I tensorflow_serving/model_servers/server_core.cc:460] Adding/updating models.
2021-05-21 10:55:08.579505: I tensorflow_serving/model_servers/server_core.cc:571]  (Re-)adding model: test
2021-05-21 10:55:08.679811: I tensorflow_serving/core/basic_manager.cc:739] Successfully reserved resources to load servable {name: test version: 1}
2021-05-21 10:55:08.679848: I tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: test version: 1}
2021-05-21 10:55:08.679856: I tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: test version: 1}
2021-05-21 10:55:08.679862: I tensorflow_serving/servables/alphafm_softmax/alphafm_softmax_bundle.cc:26] Call the constructor AlphafmSoftmaxBundle().
[10:55:08] WARNING: /tmp/xgboost.f9ak1c/src/learner.cc:851: Loading model from XGBoost < 1.0.0, consider saving it again for improved compatibility
2021-05-21 10:55:08.719028: I tensorflow_serving/servables/alphafm_softmax/alphafm_softmax_model.cc:138] Call the constructor AlphafmSoftmaxModel().
2021-05-21 10:55:08.719045: I tensorflow_serving/servables/alphafm_softmax/alphafm_softmax_model.cc:158] Call the method LoadModel(model_path).
2021-05-21 10:55:08.728887: I tensorflow_serving/servables/alphafm/feature_mapping.cc:28] Call the constructor FeatureMapping().
2021-05-21 10:55:08.728910: I tensorflow_serving/servables/alphafm/feature_mapping.cc:47] Call the method LoadModel(model_path).
2021-05-21 10:55:08.738436: W tensorflow_serving/servables/alphafm/feature_mapping.cc:57] Empty line
2021-05-21 10:55:08.738473: I tensorflow_serving/core/loader_harness.cc:87] Successfully loaded servable version {name: test version: 1}
I0521 10:55:08.762590 21850 tensorflow_serving/model_servers/server.cc:279] Running gRPC ModelServer at 0.0.0.0:8500 ...
I0521 10:55:08.771415 21850 tensorflow_serving/model_servers/server.cc:299] Exporting HTTP/REST API at:localhost:8501 ...
[evhttp_server.cc : 238] NET_LOG: Entering the event loop ...
I0521 10:55:08.808052 21850 external/brpc/src/brpc/server.cpp:1045] Server[DummyServerOf(./bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server)] is serving on port=9999.
I0521 10:55:08.808276 21850 external/brpc/src/brpc/server.cpp:1048] Check out http://haoziyu-workstation:9999 in web browser.
  ```

## 测试 XGBoost Serving model server

如果仅使用 XGBoost 模型，运行以下命令生成 `xgboost_client_cc` 可执行文件：

  ```
./tools/run_in_docker.sh bazel build -c opt //tensorflow_serving/example:xgboost_client_cc
  ```

运行以下命令测试 model server:

  ```
./tools/run_in_docker.sh ./bazel-bin/tensorflow_serving/example/xgboost_client_cc
  ```

测试命令运行完毕后，会显示如下输出：

  ```
call predict ok
outputs size is 1
the result tensor[0] is:
54 57 42 59 42 58 39 41 52 41 53 58 47 58 54 60 36 57 60 50 40 41 54 57 48 43 31 58 60 31 62 47 47 52 58 40 44 53 54 53 42 31 40 36 56 54 61 60 31 51 61 48 42 59 31 51 33 62 49 60 38 52 35 44 40 56 32 58 61 51 58 52 46 22 33 33 31 57 33 56 49 39 56 52 49 61 36 49 48 39 31 44 31 46 32 33 56 56 40 53 51 31 31 31 62 39 39 38 35 56
Done.
  ```

如果使用 XGBoost 和 FM 模型，运行以下命令生成 `alphafm_client_cc` 可执行文件：

  ```
./tools/run_in_docker.sh bazel build -c opt //tensorflow_serving/example:alphafm_client_cc
  ```

运行以下命令测试 model server:

  ```
./tools/run_in_docker.sh ./bazel-bin/tensorflow_serving/example/alphafm_client_cc
  ```

测试命令运行完毕后，会显示如下输出：

  ```
call predict ok
outputs size is 1
the result tensor[0] is:
0.0749459267
Done.
  ```

如果使用 XGBoost 和 alphaFM_softmax 模型，运行以下命令生成 `alphafm_softmax_client_cc` 可执行文件：

  ```
./tools/run_in_docker.sh bazel build -c opt //tensorflow_serving/example:alphafm_softmax_client_cc
  ```

运行以下命令测试 model server:

  ```
./tools/run_in_docker.sh ./bazel-bin/tensorflow_serving/example/alphafm_softmax_client_cc
  ```

测试命令运行完毕后，会显示如下输出：

  ```
call predict ok
outputs size is 1
the result tensor[0] is:
[0.18794255...]...
Done.
  ```
