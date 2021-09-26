# 构建 XGBoost Serving

## 前置依赖

在构建并使用 XGBoost Serving 之前, 需要安装并配置前置依赖。

### 依赖包

XGBoost Serving 是在 Ubuntu 16.04 平台开发及测试的, 在终端执行以下命令安装依赖包:

  ```
apt-get update && apt-get install -y --no-install-recommends \
        automake \
        build-essential \
        ca-certificates \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libpng-dev \
        libtool \
        libzmq3-dev \
        mlocate \
        openjdk-8-jdk\
        openjdk-8-jre-headless \
        pkg-config \
        python-dev \
        software-properties-common \
        swig \
        unzip \
        wget \
        zip \
        zlib1g-dev \
        iproute2 \
        linux-tools-generic \
        xdot \
        ghostscript \
        libc-dbg \
        ssh \
        openssl \
        libssl-dev
  ```

### Bazel

XGBoost Serving 使用 Bazel 管理构建过程，需要 0.24.1 或更高版本。安装过程如下：

1. 从 [这里](https://github.com/bazelbuild/bazel/releases) 下载 Bazel 安装包，假定下载的安装包为 bazel-0.24.1-installer-linux-x86_64.sh，执行以下命令安装 Bazel:

  ```
chmod +x bazel-0.24.1-installer-linux-x86_64.sh
./bazel-0.24.1-installer-linux-x86_64.sh --user
  ```

2. 在 ~/.bashrc 中配置 PATH 环境变量：

  ```
export PATH="$HOME/bin:$PATH"
  ```

Bazel 详细安装指南请参考：[安装指南](https://docs.bazel.build/versions/4.0.0/install.html).

### CMake

XGBoost Serving 使用 CMake 管理 XGBoost 的构建过程，需要 3.13 或更高版本。从 [这里](https://github.com/Kitware/CMake/releases) 下载 CMake 安装包。假定下载的安装包为 cmake-3.18.0-Linux-x86_64.sh，执行以下命令安装 CMake:

  ```
chmod +x cmake-3.18.0-Linux-x86_64.sh
./cmake-3.18.0-Linux-x86_64.sh --skip-license --prefix=/usr
  ```

## 构建

### Clone XGBoost Serving 仓库

执行以下命令 Clone XGBoost Serving 仓库：

  ```
git clone https://github.com/iqiyi/xgboost-serving.git
cd xgboost-serving
  ```

该命令会安装 xgboost 分支，如果需要安装其它分支，请 checkout 到所需分支。

### 构建 tensorflow_model_server

执行以下命令构建 `tensorflow_model_server`:

  ```
bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server
  ```

`tensorflow_model_server` 可执行文件放在 `bazel-bin` 路径，执行以下命令运行 `tensorflow_model_server`:

  ```
./bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
  ```

### 构建 Python API PIP 包

执行以下命令生成 `build_pip_package` 可执行文件：

  ```
bazel build -c opt //tensorflow_serving/tools/pip_package:build_pip_package
  ```

`build_pip_package` 可执行文件放在 `bazel-bin` 路径，执行以下命令生成 Python API PIP 包：

  ```
./bazel-bin/tensorflow_serving/tools/pip_package/build_pip_package $(pwd)
  ```

该命令会在当前路径生成 `tensorflow_serving_api-2.0.0-py2.py3-none-any.whl` 包，安装后可使用 tensorflow serving apis。
