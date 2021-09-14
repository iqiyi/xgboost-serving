# 在 Docker 中构建 XGBoost Serving

为了统一构建环境，我们推荐在 Docker 中构建 XGBoost Serving。XGBoost Serving Docker 开发镜像集成了构建 XGBoost Serving 所需的全部依赖。

**注意：目前我们仅支持构建运行在 Linux 平台上的可执行文件。**

## 安装 Docker

安装 Docker 的方法请参考 [安装 Docker](https://docs.docker.com/get-docker/).

## Clone XGBoost Serving 仓库

安装 Docker 后，执行以下命令 Clone XGBoost Serving 仓库：

  ```
git clone https://github.com/iqiyi/xgboost-serving.git
cd xgboost-serving
  ```

该命令会安装 xgboost 分支，如果需要安装其它分支，请 checkout 到所需分支。

## Pull Docker 开发镜像

执行以下命令 Pull Docker 开发镜像：

  ```
docker pull hzy001/xgboost-serving:latest-devel
  ```

## 构建 tensorflow_model_server

执行以下命令构建 `tensorflow_model_server`:

  ```
./tools/run_in_docker.sh bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server
  ```

`tensorflow_model_server` 可执行文件放在 `bazel-bin` 路径，执行以下命令运行 `tensorflow_model_server`:

  ```
./tools/run_in_docker.sh ./bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
  ```

## 构建 Python API PIP 包

执行以下命令生成 `build_pip_package` 可执行文件：

  ```
./tools/run_in_docker.sh bazel build -c opt //tensorflow_serving/tools/pip_package:build_pip_package
  ```

`build_pip_package` 可执行文件放在 `bazel-bin` 路径，执行以下命令生成 Python API PIP 包：

  ```
./tools/run_in_docker.sh ./bazel-bin/tensorflow_serving/tools/pip_package/build_pip_package $(pwd)
  ```

该命令会在当前路径生成 `tensorflow_serving_api-2.0.0-py2.py3-none-any.whl` 包，安装后可使用 tensorflow serving apis。
