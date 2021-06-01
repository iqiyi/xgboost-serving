[![Build](https://github.com/hzy001/xgboost-serving/actions/workflows/build.yaml/badge.svg)](https://github.com/hzy001/xgboost-serving/actions/workflows/build.yaml)
[![Test](https://github.com/hzy001/xgboost-serving/actions/workflows/test.yaml/badge.svg)](https://github.com/hzy001/xgboost-serving/actions/workflows/test.yaml)

# XGBoost Serving

This is a fork of TensorFlow Serving, extended with the support for [XGBoost](https://github.com/dmlc/xgboost), [alphaFM](https://github.com/CastellanZhang/alphaFM) and [alphaFM_softmax](https://github.com/CastellanZhang/alphaFM_softmax) frameworks. For more information about TensorFlow Serving, switch to the `master` branch or visit the [TensorFlow Serving website](https://github.com/tensorflow/serving).

----
XGBoost Serving is a flexible, high-performance serving system for
XGBoost && FM models, designed for production environments. It deals with
the *inference* aspect of XGBoost && FM models, taking models after *training* and
managing their lifetimes, providing clients with versioned access via
a high-performance, reference-counted lookup table.
XGBoost Serving derives from TensorFlow Serving and is used widely inside iQIYI.

To note a few features:

-   Can serve multiple models, or multiple versions of the same model
    simultaneously
-   Exposes gRPC inference endpoints
-   Allows deployment of new model versions without changing any client code
-   Supports canarying new versions and A/B testing experimental models
-   Adds minimal latency to inference time due to efficient, low-overhead
    implementation
-   Supports XGBoost servables, XGBoost && FM servables and XGBoost && alphaFM_Softmax servables
-   Supports computation latency distribution statistics

## Documentation

### Set up

The easiest and most straight-forward way of building and using XGBoost Serving
is with Docker images. We highly recommend this route unless you have specific
needs that are not addressed by running in a container.

*   [Build XGBoost Serving from Source](tensorflow_serving/docs/en/building.md)
*   [Build XGBoost Serving from Source with Docker](tensorflow_serving/docs/en/building_with_docker.md)*(Recommended)*

### Use

#### Export your XGBoost && FM model

In order to serve a XGBoost && FM model, simply export your XGBoot model, leaf
mapping and FM model.

Please refer to [Export XGBoost && FM model](tensorflow_serving/docs/en/export_model.md)
for details about the models's specification and how to export XGBoost && FM model.

#### Configure and Use XGBoost Serving

* [Follow a tutorial on Serving XGBoost && FM models](tensorflow_serving/docs/en/serving_basic.md)
* [Configure XGBoost Serving to make it fit your serving use case](tensorflow_serving/docs/en/serving_config.md)

### Extend

XGBoost Serving derives from TensorFlow Serving and thanks to Tensorflow Serving's highly modular architecture. You can use some parts
individually and/or extend it to serve new use cases.

* [Ensure you are familiar with building Tensorflow Serving](tensorflow_serving/g3doc/building_with_docker.md)
* [Learn about Tensorflow Serving's architecture](tensorflow_serving/g3doc/architecture.md)
* [Explore the Tensorflow Serving C++ API reference](https://www.tensorflow.org/tfx/serving/api_docs/cc/)
* [Create a new type of Servable](tensorflow_serving/g3doc/custom_servable.md)
* [Create a custom Source of Servable versions](tensorflow_serving/g3doc/custom_source.md)

## Contribute


**If you'd like to contribute to XGBoost Serving, be sure to review the
[contribution guidelines](CONTRIBUTING.md).**


## Feedback and Getting involved

* Report bugs, ask questions or give suggestions by [Github
  Issues](https://github.com/hzy001/xgboost-serving/issues)
