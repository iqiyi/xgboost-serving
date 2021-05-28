# Developing XGBoost Serving with Docker

The recommended approach to build XGBoost Serving from source is to use Docker. The XGBoost Serving Docker development images encapsulate all the dependencies you need to build your own version of XGBoost Serving.

**Note: Currently we only support building binaries that run on Linux.**

## Installing Docker

General installation instructions are [on the Docker site](https://docs.docker.com/get-docker/).

## Clone the source

After installing Docker, we need to get the source we want to build from. We will use Git to clone the xgboost branch of XGBoost Serving:

```
git clone https://github.com/hzy001/xgboost-serving.git
cd xgboost-serving
```

## Pulling a development image

For a development environment where you can build XGBoost Serving, you can try:

```
docker pull hzy001/xgboost-serving:latest-devel
```

## Building the model server

Run the following command to generate the `tensorflow_model_server` executable:

```
./tools/run_in_docker.sh bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server
```

Binaries are placed in the `bazel-bin` directory, and can be run using a command like:

```
./tools/run_in_docker.sh ./bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
```

## Building the Python API PIP package

Run the following command to generate the `build_pip_package` executable:

```
./tools/run_in_docker.sh bazel build -c opt //tensorflow_serving/tools/pip_package:build_pip_package
```

Binaries are placed in the `bazel-bin` directory, and you can run the following command to generate the Python API PIP package:

```
./tools/run_in_docker.sh ./bazel-bin/tensorflow_serving/tools/pip_package/build_pip_package $(pwd)
```

If the command finishes successfully, you will see the `tensorflow_serving_api-2.0.0-py2.py3-none-any.whl` package in the current directory and you can install this package to use the tensorflow serving apis.
