# Building XGBoost Serving from source

## Prerequisites

To build and use XGBoost Serving, you need to set up some prerequisites.

### Packages

We develop XGBoost Serving under Ubuntu 16.04, so execute the following command to install the required packages:

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

XGBoost Serving requires Bazel 0.24.1 or higher. You can find the Bazel installation instructions [here](https://docs.bazel.build/versions/4.0.0/install.html).

If you have prerequisites for Bazel, those instructions consist of the steps:

1. Download the relevant binary from [here](https://github.com/bazelbuild/bazel/releases). Let's say you download bazel-0.24.1-installer-linux-x86_64.sh. You would execute:

```
chmod +x bazel-0.24.1-installer-linux-x86_64.sh
./bazel-0.24.1-installer-linux-x86_64.sh --user
```

2. Set up your environment. Put this in your ~/.bashrc.

```
export PATH="$PATH:$HOME/bin"
```

### CMake

XGBoost Serving requires CMake 3.13 or higher to build XGBoost. You can find the CMake binary from [here](https://github.com/Kitware/CMake/releases). Suppose you download cmake-3.18.0-Linux-x86_64.sh, run the following command to install CMake:

```
chmod +x cmake-3.18.0-Linux-x86_64.sh
./cmake-3.18.0-Linux-x86_64.sh --skip-license --prefix=/usr
```

## Building

### Clone the XGBoost Serving repository

```
git clone https://github.com/hzy001/xgboost-serving.git
cd xgboost-serving
```

Note that these instructions will install the latest xgboost branch of XGBoost Serving. If you want to install a specific branch(such as a release branch), just checkout the relevant branch.

### Building the model server

To build the `tensorflow_model_server`, run the following command:

```
bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server
```

Binaries are placed in the `bazel-bin` directory, and can be run using a command like:

```
./bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
```

### Building the Python API PIP package

Run the following command to generate the `build_pip_package` executable:

```
bazel build -c opt //tensorflow_serving/tools/pip_package:build_pip_package
```

Binaries are placed in the `bazel-bin` directory, and you can run the following command to generate the Python API PIP package:

```
./bazel-bin/tensorflow_serving/tools/pip_package/build_pip_package $(pwd)
```

If the command finishes successfully, you will see the `tensorflow_serving_api-2.0.0-py2.py3-none-any.whl` package in the current directory and you can install this package to use the tensorflow serving apis.
